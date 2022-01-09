{'training': True, '_parameters': OrderedDict(), '_buffers': OrderedDict(), '_non_persistent_buffers_set': set(), '_backward_hooks': OrderedDict(), '_is_full_backward_hook': None, '_forward_hooks': OrderedDict(), '_forward_pre_hooks': OrderedDict(), '_state_dict_hooks': OrderedDict(), '_load_state_dict_pre_hooks': OrderedDict(), '_modules': OrderedDict([('module', GPT2ModelPipe(
  (tied_modules): ModuleDict()
  (0): EmbeddingPipe(
    (word_embeddings): VocabParallelEmbedding()
    (embedding_dropout): Dropout(p=0.0, inplace=False)
  )
  (2): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (3): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (4): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (5): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (6): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (7): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (8): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (9): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (10): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (11): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (12): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (13): ParallelTransformerLayerPipe(
    (input_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (attention): ParallelSelfAttention(
      (query_key_value): ColumnParallelLinear()
      (rotary_emb): RotaryEmbedding()
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.0, inplace=False)
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (15): NormPipe(
    (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (16): ParallelLinearPipe(
    (final_linear): RowParallelLinear()
  )
))]), 'dont_change_device': False, 'client_optimizer': FusedAdam (
Parameter Group 0
    betas: [0.9, 0.999]
    bias_correction: True
    eps: 1e-08
    lr: 0.0
    step: 1
    weight_decay: 0.0
), 'client_model_parameters': None, 'client_lr_scheduler': <megatron.learning_rates.AnnealingLR object at 0x7fe5f6ace250>, 'training_data': None, 'collate_fn': None, 'mpu': <deepspeed.runtime.pipe.topology.PipelineParallelGrid object at 0x7fe521a9baf0>, 'data_parallel_group': <torch._C._distributed_c10d.ProcessGroupNCCL object at 0x7fe521ac32b0>, 'global_steps': 0, 'global_samples': 0, 'micro_steps': 0, 'skipped_steps': 0, 'gradient_average': True, 'warn_unscaled_loss': True, 'config_params': {'train_batch_size': 32, 'train_micro_batch_size_per_gpu': 4, 'optimizer': {'type': 'Adam', 'params': {'lr': 0.0006, 'betas': [0.9, 0.999], 'eps': 1e-08}}, 'fp16': {'enabled': True, 'loss_scale': 0, 'loss_scale_window': 1000, 'hysteresis': 2, 'min_loss_scale': 1}, 'gradient_clipping': 1.0, 'zero_optimization': {'stage': 0, 'allgather_partitions': True, 'allgather_bucket_size': 500000000, 'overlap_comm': True, 'reduce_scatter': True, 'reduce_bucket_size': 500000000, 'contiguous_gradients': True, 'cpu_offload': False}, 'wall_clock_breakdown': True}, 'loaded_checkpoint_mp_world_size': None, 'loaded_checkpoint_dp_world_size': None, 'enable_backward_allreduce': False, 'progressive_layer_drop': None, 'dist_backend': 'nccl', 'store_gradients': False, 'store_gradients_cpu': False, 'stored_gradients': None, 'local_rank': 0, '_config': <deepspeed.runtime.config.DeepSpeedConfig object at 0x7fe5f6ace280>, 'device': device(type='cuda', index=0), 'world_size': 8, 'global_rank': 0, 'dp_world_size': 8, 'mp_world_size': 1, 'broadcast_src_rank': 0, 'timers': <deepspeed.utils.timer.SynchronizedWallClockTimer object at 0x7fe5f6ace2b0>, 'tput_timer': <deepspeed.utils.timer.ThroughputTimer object at 0x7fe5f6ace3d0>, 'training_dataloader': None, 'optimizer': FusedAdam (
Parameter Group 0
    betas: [0.9, 0.999]
    bias_correction: True
    eps: 1e-08
    lr: 0.0
    step: 1
    weight_decay: 0.0
), 'lr_scheduler': <megatron.learning_rates.AnnealingLR object at 0x7fe5f6ace250>, 'csr_tensor_module_names': set(), 'save_non_zero_checkpoint': True, 'save_zero_checkpoint': False, 'flatten': <built-in method flatten of PyCapsule object at 0x7fe5f4fd4210>, 'unflatten': <built-in method unflatten of PyCapsule object at 0x7fe5f4fd4240>, 'layer_outputs': {}, 'layers_to_hook': [], 'hooks': [<torch.utils.hooks.RemovableHandle object at 0x7fe5f6ace5b0>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f6ace610>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd41c0>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd4130>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd4280>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd42e0>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd4340>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd43a0>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd4400>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd4460>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd44c0>, <torch.utils.hooks.RemovableHandle object at 0x7fe5f4fd4520>], 'layer_name_pattern': re.compile('transformerlayer', re.IGNORECASE), 'eval_return_logits': False, 'outputs': None, 'pipeline_enable_backward_allreduce': True, 'log_batch_step_id': -1, 'micro_batch_size': 4, 'micro_batches': 1, 'grid': <deepspeed.runtime.pipe.topology.PipelineParallelGrid object at 0x7fe521a9baf0>, 'num_stages': 1, 'stage_id': 0, 'prev_stage': -1, 'next_stage': 1, 'data_iterator': None, 'batch_fn': functools.partial(<function get_batch_pipe at 0x7fe522179a60>, neox_args=NeoXArgs(distributed_backend='nccl', local_rank=0, rank=0, lazy_mpu_init=False, short_seq_prob=0.1, eod_mask_loss=False, adlr_autoresume=False, adlr_autoresume_interval=1000, seed=1234, onnx_safe=False, deepscale=False, deepscale_config=None, deepspeed_mpi=False, user_script='train.py', iteration=0, do_train=None, do_valid=None, do_test=None, global_num_gpus=8, text_gen_type=None, temperature=0.0, top_p=0.0, top_k=0, maximum_tokens=64, sample_input_file=None, sample_output_file=None, num_samples=0, recompute=False, eval_results_prefix='', eval_tasks=None, use_wandb=False, wandb_group='APh5SeLAbgmVFGPUApw6jr_1nf9dfc4', wandb_team=None, wandb_project='neox', wandb_host='https://api.wandb.ai', git_hash='0fb132c', log_dir='logs', tensorboard_dir='tensorboard', log_interval=100, log_param_norm=False, log_grad_norm=False, log_optimizer_states=False, log_gradient_noise_scale=False, gradient_noise_scale_n_batches=5, gradient_noise_scale_cpu_offload=False, pipe_parallel_size=1, model_parallel_size=1, pipe_partition_method='type:transformer|mlp', world_size=8, is_pipe_parallel=True, data_path='data/enron/enron_text_document', train_data_paths=None, test_data_paths=None, valid_data_paths=None, train_data_weights=None, valid_data_weights=None, test_data_weights=None, weight_by_num_documents=False, weighted_sampler_alpha=0.3, data_impl='mmap', mmap_warmup=False, save='checkpoints', config_files={'small.yml': '# GPT-2 pretraining setup\n{\n   # parallelism settings ( you will want to change these based on your cluster setup, ideally scheduling pipeline stages\n   # across the node boundaries )\n   "pipe-parallel-size": 1,\n   "model-parallel-size": 1,\n\n   # model settings\n   "num-layers": 12,\n   "hidden-size": 768,\n   "num-attention-heads": 12,\n   "seq-length": 2048,\n   "max-position-embeddings": 2048,\n   "norm": "layernorm",\n   "pos-emb": "rotary",\n   "no-weight-tying": true,\n\n   # these should provide some speedup but takes a while to build, set to true if desired\n   "scaled-upper-triang-masked-softmax-fusion": false,\n   "bias-gelu-fusion": false,\n\n\n   # optimizer settings\n   "optimizer": {\n     "type": "Adam",\n     "params": {\n       "lr": 0.0006,\n       "betas": [0.9, 0.999],\n       "eps": 1.0e-8,\n     }\n   },\n   "zero_optimization": {\n    "stage": 0,\n    "allgather_partitions": True,\n    "allgather_bucket_size": 500000000,\n    "overlap_comm": True,\n    "reduce_scatter": True,\n    "reduce_bucket_size": 500000000,\n    "contiguous_gradients": True,\n    "cpu_offload": False\n  },\n\n   # batch / data settings\n   "train_micro_batch_size_per_gpu": 4,\n   "data-impl": "mmap",\n   "split": "949,50,1",\n\n   # activation checkpointing\n   "checkpoint-activations": true,\n   "checkpoint-num-layers": 1,\n   "partition-activations": true,\n   "synchronize-each-layer": true,\n\n   # regularization\n   "gradient_clipping": 1.0,\n   "weight-decay": 0.0,\n   "hidden-dropout": 0.0,\n   "attention-dropout": 0.0,\n\n   # precision settings\n   "fp16": { \n     "enabled": true,\n     "loss_scale": 0,\n     "loss_scale_window": 1000,\n     "hysteresis": 2,\n     "min_loss_scale": 1\n   },\n\n   # misc. training settings\n   "train-iters": 320000,\n   "lr-decay-iters": 320000,\n   "distributed-backend": "nccl",\n   "lr-decay-style": "cosine",\n   "warmup": 0.01,\n   "save-interval": 10000,\n   "eval-interval": 1000,\n   "eval-iters": 10,\n\n   # logging\n   "log-interval": 100,\n   "steps_per_print": 10,\n   "keep-last-n-checkpoints": 4,\n   "wall_clock_breakdown": true,\n}\n', 'local_setup.yml': '# Suggested data paths when using GPT-NeoX locally\n{\n  "data-path": "data/enron/enron_text_document",\n  \n  # or for weighted datasets: \n  # "train-data-paths": ["data/enron/enron_text_document", "data/enron/enron_text_document"],\n  # "test-data-paths": ["data/enron/enron_text_document", "data/enron/enron_text_document"],\n  # "valid-data-paths": ["data/enron/enron_text_document", "data/enron/enron_text_document"],\n  # "train-data-weights": [1., 2.],\n  # "test-data-weights": [2., 1.],\n  # "valid-data-weights": [0.5, 0.4],\n\n  # If weight_by_num_documents is True, Builds dataset weights from a multinomial distribution over groups of data according to the number of documents in each group. \n  # WARNING: setting this to True will override any user provided weights\n  # "weight_by_num_documents": false,\n  # "weighted_sampler_alpha": 0.3,\n\n  "vocab-file": "data/gpt2-vocab.json",\n  "merge-file": "data/gpt2-merges.txt",\n\n  "save": "checkpoints",\n  "load": "checkpoints",\n  "checkpoint_validation_with_forward_pass": False,\n  \n  "tensorboard-dir": "tensorboard",\n  "log-dir": "logs",\n  "use_wandb": false,\n  "wandb_host": "https://api.wandb.ai",\n  "wandb_project": "neox"\n}'}, load='checkpoints', checkpoint_validation_with_forward_pass=False, save_interval=10000, no_save_optim=False, no_save_rng=False, no_load_optim=False, no_load_rng=False, finetune=False, batch_size=4, train_iters=320000, eval_iters=10, keep_last_n_checkpoints=4, eval_interval=1000, split='949,50,1', vocab_file='data/gpt2-vocab.json', merge_file='data/gpt2-merges.txt', num_workers=2, exit_interval=None, attention_dropout=0.0, hidden_dropout=0.0, weight_decay=0.0, checkpoint_activations=True, checkpoint_num_layers=1, deepspeed_activation_checkpointing=True, contiguous_checkpointing=False, checkpoint_in_cpu=False, synchronize_each_layer=True, profile_backward=False, partition_activations=True, gas=1, clip_grad=1.0, hysteresis=2, dynamic_loss_scale=True, loss_scale=None, loss_scale_window=1000.0, min_scale=1.0, char_level_ppl=False, tokenizer_type='GPT2BPETokenizer', padded_vocab_size=50304, optimizer_type='Adam', use_bnb_optimizer=False, zero_stage=0, zero_reduce_scatter=True, zero_contiguous_gradients=True, zero_reduce_bucket_size=500000000, zero_allgather_bucket_size=500000000, lr=0.0006, lr_decay_style='cosine', lr_decay_iters=320000, min_lr=0.0, warmup=0.01, override_lr_scheduler=False, use_checkpoint_lr_scheduler=False, precision='fp16', num_layers=12, hidden_size=768, num_attention_heads=12, seq_length=2048, max_position_embeddings=2048, norm='layernorm', layernorm_epsilon=1e-05, rms_norm_epsilon=1e-08, scalenorm_epsilon=1e-08, pos_emb='rotary', rpe_num_buckets=32, rpe_max_distance=128, no_weight_tying=True, attention_config=['global', 'global', 'global', 'global', 'global', 'global', 'global', 'global', 'global', 'global', 'global', 'global'], sparsity_config={}, num_unique_layers=None, param_sharing_style='grouped', make_vocab_size_divisible_by=128, activation='gelu', scaled_upper_triang_masked_softmax_fusion=False, scaled_masked_softmax_fusion=False, bias_gelu_fusion=False, bias_dropout_fusion=False, fp16_lm_cross_entropy=False, init_method_std=0.02, apply_query_key_layer_scaling=False, use_cpu_initialization=False, attention_softmax_in_fp32=False, rotary_pct=1.0, rotary_emb_base=10000, init_method='normal', output_layer_init_method='scaled_normal', gmlp_attn_dim=64, gpt_j_residual=False, soft_prompt_tuning=None, output_layer_parallelism='row', deepspeed=True, train_batch_size=32, train_micro_batch_size_per_gpu=4, gradient_accumulation_steps=1, optimizer={'type': 'Adam', 'params': {'lr': 0.0006, 'betas': [0.9, 0.999], 'eps': 1e-08}}, scheduler=None, fp32_allreduce=False, prescale_gradients=False, gradient_predivide_factor=1.0, sparse_gradients=False, fp16={'enabled': True, 'loss_scale': 0, 'loss_scale_window': 1000, 'hysteresis': 2, 'min_loss_scale': 1}, amp=None, gradient_clipping=1.0, zero_optimization={'stage': 0, 'allgather_partitions': True, 'allgather_bucket_size': 500000000, 'overlap_comm': True, 'reduce_scatter': True, 'reduce_bucket_size': 500000000, 'contiguous_gradients': True, 'cpu_offload': False}, steps_per_print=10, wall_clock_breakdown=True, dump_state=False, flops_profiler=None, zero_allow_untested_optimizer=False, hostfile=None, include=None, exclude=None, num_nodes=-1, num_gpus=None, master_port=29500, master_addr=None, launcher='pdsh', detect_nvlink_pairs=False)), '_force_grad_boundary': False, 'batch_timer': <deepspeed.utils.timer.ThroughputTimer object at 0x7fe5f4fd4580>, 'is_pipe_parallel': False, 'is_data_parallel': True, 'is_model_parallel': False, 'is_pipe_partitioned': False, 'is_grad_partitioned': False, 'num_pipe_buffers': 0, 'pipe_buffers': {'inputs': [], 'labels': [], 'outputs': [], 'output_tensors': []}, 'pipe_recv_buf': None, 'grad_layer': None, 'meta_buffer': None, 'first_output_send': True, 'first_gradient_send': True, 'timer_values': None, 'loss': tensor(0., device='cuda:0'), 'total_loss': None, 'agg_loss': tensor(0., device='cuda:0'), 'dp_group_loss': tensor(0., device='cuda:0'), 'loss_model': functools.partial(<function cross_entropy at 0x7fe58f7cfdc0>, _fp16=False), 'has_attention_mask': True, 'total_params': 162322944}