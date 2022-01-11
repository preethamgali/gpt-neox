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

from training import get_model

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

def setup_model_and_optimizer_to_distil(distil_neox_args, iteration=None):

    distil_neox_args.set_student()
    distil_neox_args.update_value("load", distil_neox_args.load_student)
    student_model = get_model(neox_args=distil_neox_args, inference=False, get_key_value=False)
    assert distil_neox_args.load is not None, "Please provide the teacher model load path for distillation"
    _ = load_checkpoint(neox_args=distil_neox_args,model=student_model)

    distil_neox_args.set_teacher()
    distil_neox_args.update_value("load", distil_neox_args.load_teacher)
    teacher_model = get_model(neox_args=distil_neox_args, inference=False, get_key_value=False)
    if distil_neox_args.load is not None:
        _ = load_checkpoint(neox_args=distil_neox_args,model=teacher_model)

    print_rank_0("student_model", student_model)
    print_rank_0("teacher_model", teacher_model)
    return None, None, None

