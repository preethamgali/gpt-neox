"""Distil"""
from megatron.neox_arguments import NeoXArgsDistillation
from megatron.distilling import pretrain_by_distil
from megatron.distilation_modules import DistilDecorator

if __name__ == "__main__":
    distil_neox_args = NeoXArgsDistillation.consume_neox_args()
    distil_neox_args.configure_distributed_args()
    distil_neox_args.build_tokenizer() # tokenizer needs to be build in training in order to set the padding vocab
    distil_neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    DistilDecorator.do_distillation = distil_neox_args.do_distillation
    pretrain_by_distil(distil_neox_args=distil_neox_args)