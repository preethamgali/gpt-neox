# coding=utf-8
#
# Copyright 2021 Biderman et al. This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

import torch
import torch.nn as nn
from collections import defaultdict

from functools import partial
from megatron.model.utils import Lambda, SequentialWrapper, recursive_setattr
from megatron.model.norms import get_norm
from megatron.model.init_functions import get_init_methods

from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
from megatron.model.transformer import ParallelTransformerLayerPipe, NormPipe, ParallelLinearPipe, parallel_lm_logits, ParallelLinear
from megatron.model.gmlp import GMLPBlock
from megatron.model.word_embeddings import EmbeddingPipe, SoftEmbedding
from megatron.distilation_modules import DistilDecorator

# Pipeline parallelism
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from typing import Union, List

def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores

def get_topk_mask(teacher_logits, topk=1024):
    device = teacher_logits.device
    batch_size, seq_len, hidden_size = teacher_logits.shape
    start_indicies = torch.arange(batch_size*seq_len) * hidden_size

    top_k_indicies = torch.topk(teacher_logits, topk, sorted=False)[1]
    top_k_indicies = top_k_indicies.to(device).view(-1,topk) + start_indicies.to(device).unsqueeze(-1).expand(-1,topk)
    top_k_indicies = top_k_indicies.flatten()

    mask = torch.zeros_like(teacher_logits, dtype=torch.bool).to(device).flatten()
    mask[top_k_indicies.long()] = True
    mask = mask.view(batch_size, seq_len, hidden_size)
    return mask, topk

def topk_kldiv(output, labels, topk):

    batch_size, seq_len, hidden_size = output.shape
    soft_output = torch.nn.Softmax(dim=2)(output)
    soft_labels = torch.nn.Softmax(dim=2)(labels)

    mask, topk = get_topk_mask(soft_labels, topk)
    masked_logits_shape = (-1, topk)

    masked_soft_output = soft_output[mask].view(*masked_logits_shape)
    masked_soft_output = masked_soft_output.add(1e-5)
    masked_soft_labels = soft_labels[mask].view(*masked_logits_shape)
    masked_soft_labels = torch.nn.Softmax()(masked_soft_labels)
    losses = torch.nn.KLDivLoss(reduction='sum')(masked_soft_output.log(), masked_soft_labels)/(batch_size * seq_len)
    return losses

def kldiv_loss_fn(output, labels, topk=None, _fp16=False):

    labels, loss_mask = labels[0], labels[1]

    output = output.float().contiguous() if _fp16 else output.contiguous()
    labels = labels.float().contiguous() if _fp16 else labels.contiguous()
    
    if _fp16:
        assert (output.dtype == torch.half and labels.dtype == torch.half and loss_mask.dtype == torch.half)

    if topk is not None:
        #mask, topk = get_topk_mask(labels, topk)
        #masked_logits_shape = labels.shape[:-1] + (topk,)

        #losses = mpu.loss.vocab_parallel_KLDivLoss(output[mask].view(*masked_logits_shape),
        #                                           labels[mask].view(*masked_logits_shape))

        return topk_kldiv(output, labels, topk)
    else:
        losses = mpu.loss.vocab_parallel_KLDivLoss(output, labels)

    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

def mse_loss_fn(output, labels, _fp16=False):

    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert (output.dtype == torch.half and labels.dtype == torch.half and loss_mask.dtype == torch.half)
        losses = mpu.loss.vocab_parallel_MSELoss(output.contiguous(), labels.contiguous())
    else:
        losses = mpu.loss.vocab_parallel_MSELoss(output.float().contiguous(), labels.float().contiguous())
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

def combined_loss_fn(output, labels, self, alpha_lm=0, alpha_kld=0, alpha_mse=0, _fp16=False):
    labels, loss_mask = labels[0], labels[1]
    prev_output, input_args, teacher_args, student_args = output
    teacher_hidden_states ,teacher_logits = teacher_args
    student_hidden_states, student_logits = student_args
    
    del prev_output
    del input_args
    del teacher_hidden_states
    del student_hidden_states

    mse_loss = torch.tensor(0).to(student_logits)
    if alpha_mse > 0:
        mse_loss = mse_loss_fn(student_logits, (teacher_logits, loss_mask), _fp16=_fp16)
        mse_loss += alpha_mse * mse_loss

    kld_loss = torch.tensor(0).to(student_logits)
    if alpha_kld > 0:
        kld_loss = kldiv_loss_fn(student_logits, (teacher_logits, loss_mask), _fp16=_fp16)
        kld_loss += alpha_kld * kld_loss
    
    lm_loss = torch.tensor(0).to(student_logits)
    if alpha_lm > 0:
        lm_loss = cross_entropy(student_logits, (labels, loss_mask), _fp16=_fp16)
        lm_loss = alpha_lm * lm_loss
    
    loss = lm_loss + kld_loss + mse_loss
    count = torch.tensor(1).to(lm_loss.device)

    torch.distributed.all_reduce(count, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(lm_loss, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(kld_loss, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(mse_loss, group=mpu.get_data_parallel_group())

    count = count * self.neox_args.gradient_accumulation_steps
    if self._losses==None:
        self._losses = [lm_loss.clone().detach()/count,
                        kld_loss.clone().detach()/count,
                        mse_loss.clone().detach()/count]
    else:
        self._losses[0] += lm_loss.clone().detach()/count
        self._losses[1] += kld_loss.clone().detach()/count
        self._losses[2] += mse_loss.clone().detach()/count
    return loss

def cross_entropy(output, labels, _fp16=False):
    """ From pretrain_gpt2:forward_step() """
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert (output.dtype == torch.half and loss_mask.dtype == torch.half)
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

@DistilDecorator.distil_func(is_class_function=False)
def _pre_transformer_block(args):
    # used instead of a lambda layer to pass outputs of the word embedding to the transformer block
    # using a custom function means we don't have to have this _inference mode which makes everything tricky
    in_inference = len(args) == 3
    in_train = len(args) == 2
    # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
    if in_inference:
        # we need to add a container to cache `presents` from each layer's forward pass
        # inputs/outputs are now (hidden_states, layer_past, presents, attention_mask)
        fn = lambda x: (x[0].transpose(0, 1).contiguous(), x[1], torch.Tensor(), *x[2:])
    elif in_train:
        fn = lambda x: (x[0].transpose(0, 1).contiguous(), *x[1:])
    else:
        raise ValueError('Incorrect number of args in `_pre_transformer_block`')
    return fn(args)

@DistilDecorator.distil_func(is_class_function=False)
def _post_transformer_block(args):
    # used instead of a lambda layer to pass outputs of the transformer block to the final layer
    # using a custom function means we don't have to have this _inference mode which makes everything tricky
    in_inference = len(args) == 4
    in_train = len(args) == 2
    if in_inference:
        # we can get rid of the mask / pasts now
        # from (hidden_states, layer_past, presents, attention_mask)
        # to (hidden_states.T, presents)
        fn = lambda x: (x[0].transpose(0, 1).contiguous(), x[2])
    elif in_train:
        # Undo data format change and drop mask
        fn = lambda x: x[0].transpose(0, 1).contiguous()
    else:
        raise ValueError('Incorrect number of args in `_post_transformer_block`')
    return fn(args)


class GPT2ModelPipe(PipelineModule, torch.nn.Module):
    """GPT2Model adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    def __init__(self, neox_args, num_tokentypes=0, parallel_output=True, topology=None, inference=False,
                 get_key_value=True):
        self.neox_args = neox_args

        self._inference = inference
        self.get_key_value = get_key_value if inference else False
        self.parallel_output = parallel_output
        self.hidden_size = self.neox_args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(self.neox_args)
        self.embedding_type = self.neox_args.pos_emb
        self.__topology__ = topology

        self.specs = []
        self.init_specs()

        if self.neox_args.do_distillation:
            self._losses = None
            loss_fn = partial(combined_loss_fn,
                            self=self,
                            alpha_lm=self.neox_args.alpha_lm,
                            alpha_kld=self.neox_args.alpha_kld,
                            alpha_mse=self.neox_args.alpha_mse,
                            _fp16=self.neox_args.fp16_lm_cross_entropy)

        else:
            loss_fn = partial(cross_entropy, _fp16=self.neox_args.fp16_lm_cross_entropy)

        if self.neox_args.checkpoint_activations:
            interval = self.neox_args.checkpoint_num_layers
        else:
            interval = 0
        super().__init__(layers=self.specs,
                         loss_fn=loss_fn if not self._inference else None,
                         topology=topology,
                         activation_checkpoint_interval=interval,
                         partition_method=neox_args.pipe_partition_method,
                         checkpointable_layers=['GMLPBlock', 'ParallelTransformerLayerPipe'])

    def insert_layers(self, layers: Union[nn.Module, nn.ModuleList, nn.Sequential, List], idx):
        """
        inserts the layers in `layers` into the pipe model at `idx`.
        """
        if isinstance(layers, nn.Module):
            self.specs.insert(idx, layers)
        elif any([isinstance(layers, nn.ModuleList), isinstance(layers, nn.Sequential)]):
            self.specs[idx:idx] = layers
        elif isinstance(layers, list):
            assert all([hasattr(l, '__call__') or isinstance(l, LayerSpec) for l in layers]), "all items in `layers` must be Callables"
            self.specs[idx:idx] = layers
        else:
            raise ValueError(f'layer passed into {self.__class__.__name__}.insert_layer() should be either an nn.Module, an nn.ModuleList, an nn.Sequential object, or a list of callables not a {type(layers)}')
        
        # re-initialize parent class
        super().__init__(layers=self.specs,
                         loss_fn=self.loss_fn,
                         topology=self.__topology__,
                         activation_checkpoint_interval=self.activation_checkpoint_interval,
                         partition_method=self.neox_args.pipe_partition_method,
                         checkpointable_layers=['GMLPBlock', 'ParallelTransformerLayerPipe'])

    def init_specs(self):
        weight_tying = not self.neox_args.no_weight_tying
        if self.embedding_type == 'rpe':
            rpe_emb = ParallelRelativePositionBias(neox_args=self.neox_args, causal=True,
                                                   num_buckets=self.neox_args.rpe_num_buckets,
                                                   max_distance=self.neox_args.rpe_max_distance,
                                                   heads=self.neox_args.num_attention_heads)
        self.specs = []
        # Embedding layer
        # input will be (input_ids, position_ids, attention_mask) in Training
        # and (input_ids, position_ids, attention_mask, layer_past) in Inference
        if weight_tying:
            self.specs.append(TiedLayerSpec('embed',
                                            EmbeddingPipe,
                                            self.neox_args,
                                            self.hidden_size,
                                            self.neox_args.padded_vocab_size,
                                            self.neox_args.max_position_embeddings,
                                            self.neox_args.hidden_dropout,
                                            self.init_method,
                                            self.num_tokentypes,
                                            tied_weight_attr='word_embeddings_weight'))
        else:
            self.specs.append(LayerSpec(EmbeddingPipe,
                                        self.neox_args,
                                        self.hidden_size,
                                        self.neox_args.padded_vocab_size,
                                        self.neox_args.max_position_embeddings,
                                        self.neox_args.hidden_dropout,
                                        self.init_method,
                                        self.num_tokentypes))

        # NB: in inference, the attention mask always needs to be the *last* item in the args when being passed from 
        # one stage to the next, because deepspeed is hacks on top of hacks.
        #
        # outputs are now
        #           Train: (hidden_states,  attention_mask)
        #           Inference: (hidden_states, layer_past, attention_mask)

        self.specs.append(_pre_transformer_block)

        # Transformer layers
        for i in range(self.neox_args.num_layers):
            layer_type = self.neox_args.attention_config[i]
            if layer_type in ["gmlp", "amlp"]:
                self.specs.append(
                    LayerSpec(
                        GMLPBlock,
                        init_method=self.init_method,
                        layer_number=i,
                        output_layer_init_method=self.output_layer_init_method,
                        neox_args=self.neox_args,
                        mask_fn=gpt2_attention_mask_func
                    )
                )
            else:
                self.specs.append(
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        neox_args=self.neox_args,
                        attention_mask_func=gpt2_attention_mask_func,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method,
                        layer_number=i,
                        rpe=rpe_emb if self.neox_args.pos_emb == 'rpe' else None,
                        rotary=self.neox_args.pos_emb == 'rotary',
                        get_key_value=self.get_key_value
                    )
                )

        self.specs.append(_post_transformer_block)

        # NormPipe is a helper class to pass presents through to the output when doing inference
        norm, eps = get_norm(self.neox_args)
        self.specs.append(
            LayerSpec(NormPipe,
                      norm,
                      self.neox_args.hidden_size,
                      eps=eps))

        # outputs are now
        #           Train: hidden_states
        #           Inference: (hidden_states, presents)

        @DistilDecorator.distil_func(is_class_function=True)
        def _logits_helper(embedding, lm_output):
            """Just a wrapper to massage inputs/outputs from pipeline. """
            if self._inference and len(lm_output) == 2:
                hidden_states, presents = lm_output
                logits = parallel_lm_logits(
                    hidden_states,
                    embedding.word_embeddings_weight,
                    self.parallel_output)
                return logits, presents
            else:
                logits = parallel_lm_logits(
                    lm_output,
                    embedding.word_embeddings_weight,
                    self.parallel_output)
                return logits

        if weight_tying:
            self.specs.append(
                TiedLayerSpec('embed',
                              EmbeddingPipe,
                              self.neox_args,
                              self.hidden_size,
                              self.neox_args.padded_vocab_size,
                              self.neox_args.max_position_embeddings,
                              self.neox_args.hidden_dropout,
                              self.init_method,
                              self.num_tokentypes,
                              forward_fn=_logits_helper,
                              tied_weight_attr='word_embeddings_weight')
            )
        else:
            self.specs.append(
                LayerSpec(
                    ParallelLinearPipe,
                    neox_args=self.neox_args,
                    init_method=self.init_method,
                    parallel_output=self.parallel_output,
                    inference=self._inference
                )
            )
        # output in training should just be logits
        # in inference it will be (logits, presents) (assuming get_key_value) is true

    def _set_parallel_output(self, value):
        # sets the parallel output value of the final layer to value
        final_layer = list(self.forward_funcs)[-1]
        if isinstance(final_layer, (ParallelLinearPipe, ParallelLinear)):
            final_layer.final_linear.set_parallel_output(value)

    def inference_mode(self, cache=True):
        # first set caching to true if specified
        recursive_setattr(self.forward_funcs, "get_key_value", cache, assert_type=bool)
        # then set parallel output of the final layer to false so we don't have to gather the output manually
        self._set_parallel_output(False)

    
    def train_mode(self):
        # set caching to false
        recursive_setattr(self.forward_funcs, "get_key_value", False)
        # then set parallel output to true (more efficient training)
        self._set_parallel_output(True)


    def to_sequential(self):
        """
        Transforms the PipelineModule to a plain nn.Sequential module
        :return:
        """
        layers = []
        tied_layers = defaultdict(list)
        for n, spec in enumerate(self.specs):
            if isinstance(spec, TiedLayerSpec):
                if spec.key in tied_layers:
                    # receiver
                    layers.append(Lambda(lambda x: spec.forward_fn(tied_layers[spec.key][0], x)))
                else:
                    # owner
                    module = spec.build(log=False)
                    layers.append(module)
                    tied_layers[spec.key].append(module)
            elif isinstance(spec, LayerSpec):
                layers.append(spec.build(log=False))
            elif hasattr(spec, '__call__'):
                # check that it's a callable function
                layers.append(Lambda(spec))
            else:
                raise ValueError(f'Layer number {n} ({spec}) Not recognized')
        model = SequentialWrapper(layers,
                                  self.activation_checkpoint_interval,
                                  self.activation_checkpoint_func,
                                  parent_class_name=self.__class__.__name__)
        return model
