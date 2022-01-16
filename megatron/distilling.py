"""Distilling utilities."""
from datetime import datetime
from functools import partial

import math
import sys

import torch
import deepspeed
import numpy as np

from megatron.utils import (
    Timers,
    init_wandb,
    get_ltor_masks_and_position_ids,
    reduce_losses,
)


from megatron import print_rank_0, mpu
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
)
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.data.data_utils import build_train_valid_test_data_iterators
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.logging import tb_wandb_log, training_log
from megatron.utils import (
    OverflowMonitor,
    get_noise_scale_logger,
    get_total_params,
    CharCounter,
)
from megatron.model.gpt2_model import cross_entropy
from eval_tasks import run_eval_harness

from megatron.training import get_model, get_optimizer, get_learning_rate_scheduler, setup_model_and_optimizer


def pretrain_by_distil(distil_neox_args):

    distil_neox_args.set_student()

    # setup logging and timers
    init_wandb(neox_args=distil_neox_args)
    timers = Timers(
        use_wandb=distil_neox_args.use_wandb, tensorboard_writer=distil_neox_args.tensorboard_writer
    )

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(neox_args=distil_neox_args)

    # Model, optimizer, and learning rate.
    timers("model and optimizer").start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer_to_distil(
        distil_neox_args=distil_neox_args
    )
    timers("model and optimizer").stop()


def get_distil_model(distil_neox_args):
    def load_model(distil_neox_args, is_teacher):
        if is_teacher:
            distil_neox_args.set_teacher()
            load_path = distil_neox_args.load_teacher
        else:
            distil_neox_args.set_student()
            load_path = distil_neox_args.load_student

        model = get_model(neox_args=distil_neox_args,
                          inference=False, get_key_value=False)
        assert distil_neox_args.load is not None or not (is_teacher and load_path is None), \
            "Please provide the teacher model load path for distillation"
        if load_path is not None and distil_neox_args.load is None:
            model_type = 'teacher' if is_teacher else 'studnet'
            print_rank_0(
                f"Loading {model_type} model weights from {load_path}")
            model.load_state_dir(load_path)
        return model

    def load_distil_model(distil_neox_args, teacher_model, student_model):

        # if teacher output logits are provided as input we dont need teacher model
        if distil_neox_args.input_teacher_output:
            return student_model, list(student_model.named_parameters())

        # if teacher hiddent_state are provided as input we only need last layer of teacher model
        if distil_neox_args.input_teacher_hidden_state:
            teacher_model_specs = teacher_model.specs[-1:]
        else:
            teacher_model_specs = teacher_model.specs

        distil_model = load_model(distil_neox_args, is_teacher=False)
        distil_model.insert_layers(teacher_model_specs, 0)

        distil_model_layers = list(distil_model.state_dict().items())
        student_model_layers = list(student_model.state_dict().items())
        teacher_model_layers = list(teacher_model.state_dict().items())

        if distil_neox_args.input_teacher_hidden_state:
            n_teacher_model_layers = len(distil_model_layers)-len(student_model_layers)
            teacher_model_layers = teacher_model_layers[-n_teacher_model_layers:]

        if distil_neox_args.load is None:

            teacher_student_model_layers = teacher_model_layers + student_model_layers

            assert len(distil_model_layers) == len(teacher_model_layers)+len(student_model_layers), \
                f"Number of distil model layers: {len(distil_model_layers)} is not equal to" \
                f"number of teacher and student model layers combined:" \
                f"{len(teacher_model_layers)}+{len(student_model_layers)}={len(teacher_student_model_layers)}"

            from collections import OrderedDict
            new_distil_model_state_dict = OrderedDict()
            for layer_num, (distil_model_layer, teacher_student_model_layer) in \
                    enumerate(zip(distil_model_layers, teacher_student_model_layers)):

                distil_model_key, _ = distil_model_layer
                other_model_key, other_model_value = teacher_student_model_layer

                distil_layer_name = ".".join(distil_model_key.split(".")[1:])
                other_layer_name = ".".join(distil_model_key.split(".")[1:])

                assert other_layer_name == distil_layer_name, \
                    "distil layer: {distil_layer_name} is not same as combined teacher and student layer: {other_layer_name}"

                new_distil_model_state_dict[distil_model_key] = other_model_value
            distil_model.load_state_dict(new_distil_model_state_dict)

        elif distil_neox_args.load is not None:
            print_rank_0(
                f"Loading distil model weights from {distil_neox_args.load}")
            distil_model.load_state_dir(distil_neox_args.load)

        n_distil_params = len(list(distil_model.named_parameters()))
        n_student_params = len(list(student_model.named_parameters()))
        n_teacher_params = n_distil_params - n_student_params

        required_named_parameters = []
        for idx, param in enumerate(list(distil_model.named_parameters())):
            name, parameter = param
            if idx < n_teacher_params:
                parameter.requires_grad=False
            else:
                required_named_parameters.append(param)

        return distil_model, required_named_parameters

    torch.distributed.barrier()
    teacher_model = load_model(distil_neox_args, is_teacher=True)
    student_model = load_model(distil_neox_args, is_teacher=False)
    distil_model, trainable_params = load_distil_model(
        distil_neox_args, teacher_model, student_model)
    torch.distributed.barrier()

    return distil_model, trainable_params


def get_distil_optimizer(model, trainable_params, distil_neox_args):
    optimizer, param_groups = get_optimizer(model, distil_neox_args)
    print_rank_0(optimizer)
    return optimizer, param_groups


def setup_model_and_optimizer_to_distil(distil_neox_args, inference=False, get_key_value=True, iteration=None):

    distil_model, trainable_params = get_distil_model(
        distil_neox_args=distil_neox_args)
    optimizer, param_groups = get_distil_optimizer(model=distil_model,
                                                   trainable_params=trainable_params,
                                                   distil_neox_args=distil_neox_args)
    # lr_scheduler = get_learning_rate_scheduler(
    #     optimizer=optimizer, neox_args=distil_neox_args)
    return None, None, None