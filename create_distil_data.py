import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import torch
import deepspeed
import numpy as np

from megatron.training import get_batch
from megatron.utils import setup_for_inference_or_eval
from megatron.data.data_utils import make_indexed_dataset
from megatron.data.gpt2_dataset import _num_epochs, _num_tokens
from eval_tasks.eval_adapter import EvalHarnessAdapter
from megatron.utils import mpu, print_rank_0

from megatron.data.data_utils import build_train_valid_test_data_iterators

def forward_step(data_iterator, model, neox_args, timers, return_logits=False):
    """Forward step."""

    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        neox_args=neox_args, data_iterator=data_iterator
    )
    
    outputs = model((tokens, position_ids, attention_mask))

    if return_logits:
        return None, outputs

def create_data(
    neox_args, data_iterator, model, verbose=False, timers=None
):
    """Evaluation.
    neox_args: NeoX Arguments
    forward_step_fn: function with args `neox_args, timers,
                    data_iterator & model that will run a forward pass on the model
    data_iterator: Iterator that iterates over batches of data. Should return data in the form:
                    {'text': np.array([tokens], dtype=np.int64)}
                    where the size of the array is the model's context size + 1
                    (`get_batch` transforms it into inputs / labels)
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    dataset_indicies_list = []
    output_list = []


    with torch.no_grad():
        iteration = 0
        while iteration < neox_args.train_iters:
            iteration += 1
            # if verbose and iteration % neox_args.log_interval == 0:
            print_rank_0(
                "Iter {}/{}".format(iteration, neox_args.train_iters)
            )
            dataset_indicies = data_iterator._index_sampler.current_batch_idx

            # Forward evaluation
            loss, outputs = forward_step(
                model=model,
                data_iterator=data_iterator,
                neox_args=neox_args,
                timers=None,
                return_logits=True
            )
            print_rank_0(outputs.shape)
            output_list.append(outputs.cpu().detach().numpy())
            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    # Move model back to the train mode.
    model.train()

def calulaute_n_iters(neox_args):

    indexed_dataset = make_indexed_dataset(path=neox_args.data_path,
                                           impl=neox_args.data_impl)
    total_num_of_documents = indexed_dataset.sizes.shape[0]   
    sizes =  indexed_dataset.sizes
    seq_length = neox_args.seq_length
    documents = np.arange(start=0, stop=total_num_of_documents,
                          step=1, dtype=np.int32)
    tokens_per_epoch = _num_tokens(documents, sizes)
    return int(((tokens_per_epoch - 1) // seq_length) / neox_args.train_batch_size)

def main():
    print_rank_0("Considering only train dataset args")

    # setting dummy values
    args = {
        "train_iters" : -1,
        "eval_interval" : -1,
        "eval_iters" : -1,
        "split" : "1,0,0",
        "gradient_accumulation_steps" : 1,
        "pipe_parallel_size" : 0
    }
    print_rank_0(f"Resetting few neox args to : {args}")

    model, neox_args = setup_for_inference_or_eval(inference=False, get_key_value=False, overwrite_values=args)
    #making the model to retunr hidden state as ouput
    model.module.activation_checkpoint_interval = 0
    model.module.sequential = model.module.sequential[:-1]
    
    neox_args.train_iters = calulaute_n_iters(neox_args)
    train_data_iterator, _, _, = build_train_valid_test_data_iterators(neox_args=neox_args)
    create_data(neox_args, train_data_iterator, model, verbose=False, timers=None)

if __name__ == "__main__":
  main()
