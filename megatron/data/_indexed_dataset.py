# # import os
# # import re
# # import torch
# # import numpy as np

# # class MapGPT2DatasetIndex2MmapModelOutput(torch.utils.data.Dataset):
# #     """Mapping from GPT2Dataset sample index to its saved model output on memmap array index"""

# #     def __init__(self, path, seq_length, tkn_output_length):
# #         super().__init__()
# #         self._path = path
# #         self.map_indicies()

# #         self.seq_length = seq_length
# #         self.tkn_output_length = tkn_output_length

# #         self.opened_filename = None
# #         self.memmap_modeloutput = None

# #     def get_all_files(self):
# #         files_list = os.listdir(self._path)
# #         files_list = [filename.replace('.dat', '') for filename in files_list if "_all" in filename and filename.endswith('.dat')]
# #         files_list = sorted(files_list, 
# #                             key = lambda filename: int(re.findall(r'\d+', filename.split("_iter")[-1])[0]))
# #         return files_list

# #     def set_mapping(self):
# #         files_list = self.get_all_files()

# #         self.mapping_dict = {}
# #         self.n_samples_per_memmap = {}
# #         for filename in files_list:
# #             _filename = filename+".index.npy"
# #             filepath = os.path.join(self._path, _filename)
# #             dataset_index = np.load(filepath)
# #             self.n_samples_per_memmap[filename+".dat"] = dataset_index.shape[0]
# #             # iteration = int(re.findall(r'\d+', filename.split("_iter")[-1])[0])
# #             for output_index, gpt2datasetindex in enumerate(dataset_index):
# #                 self.mapping_dict[gpt2datasetindex] = (filename+".dat", output_index)

# #     def get_memmap_file(self, filename):
# #         if self.opened_filename == filename:
# #             return self.memmap_modeloutput
# #         else:
# #             del self.memmap_modeloutput
# #             model_output_file_path = os.path.join(self._path, filename)
# #             self.opened_filename = filename
# #             dtype = 'float16' if 'fp16' in filename else 'float32'
# #             tensor_shape = (self.n_samples_per_memmap[filename], self.seq_length, self.tkn_output_length)
# #             self.memmap_modeloutput = np.memmap(model_output_file_path, dtype=dtype, mode='r', shape=tensor_shape)
# #         return self.memmap_modeloutput

# #     def __getitem__(self, idx):
# #         filename, output_index = self.mapping_dict[idx]
# #         memmap_modeloutput = self.get_memmap_file(filename)
# #         model_output = memmap_modeloutput[output_index]
# #         return np.expand_dims(model_output, axis=0)

# #     def __del__(self):
# #         del self.memmap_modeloutput



# create_distil_data.py
# import os
# import re
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                              os.path.pardir)))

# import torch
# import deepspeed
# import numpy as np

# from megatron.training import get_batch
# from megatron.utils import setup_for_inference_or_eval
# from megatron.data.data_utils import make_indexed_dataset
# from megatron.data.gpt2_dataset import _num_epochs, _num_tokens
# from eval_tasks.eval_adapter import EvalHarnessAdapter
# from megatron.utils import mpu, print_rank_0

# from megatron.data.data_utils import build_train_valid_test_data_iterators

# def forward_step(data_iterator, model, neox_args):
#     """Forward step."""

#     tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
#         neox_args=neox_args, data_iterator=data_iterator
#     )
    
#     outputs = model((tokens, position_ids, attention_mask))

#     return outputs

# def numpy_memmap_file(neox_args, iteration, rank = None, batch_size = None, start_index=None, end_index=None):

#     save_dir = neox_args.distil_data_gen['save_dir']
#     save_interval = neox_args.distil_data_gen['save_interval']
#     save_hidden_state = neox_args.distil_data_gen['save_hidden_state']

#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)

#     _dtype = 'fp16' if neox_args.fp16['enabled'] else 'fp32'

#     if batch_size is None:
#         batch_size = save_interval*neox_args.train_micro_batch_size_per_gpu

#     rank = rank if rank is not None else "_all"
    
#     if save_hidden_state:
#         if rank != "_all":
#             model_output_filename = f"model_last_hidden_state_{_dtype}_rank{rank}_iter{iteration}.dat"
#         else:
#             model_output_filename = f"model_last_hidden_state_{_dtype}_rank{rank}_{start_index}_{end_index}_iter{iteration}.dat"

#         tensor_shape = (batch_size, 
#                         neox_args.seq_length, 
#                         neox_args.hidden_size)
#     else:
#         if rank != "_all":
#             model_output_filename = f"model_output_logit_{_dtype}_rank{rank}_iter{iteration}.dat"
#         else:
#             model_output_filename = f"model_output_logit_{_dtype}_rank{rank}_{start_index}_{end_index}_iter{iteration}.dat"

#         tensor_shape = (batch_size, 
#                         neox_args.seq_length, 
#                         neox_args.padded_vocab_size)

#     model_output_filename_path = os.path.join(save_dir, model_output_filename)

#     dtype = 'float16' if neox_args.fp16['enabled'] else 'float32'
#     np_memmap_array = np.memmap(model_output_filename_path, dtype=dtype, mode='w+', shape=tensor_shape)

#     return np_memmap_array, model_output_filename_path

# def save_output(neox_args,
#                 np_memmap_array, 
#                 dataset_indicies, 
#                 model_output_filename_path):

#     iteration = int(re.findall(r'\d+.dat', model_output_filename_path)[0].replace('.dat', ''))
#     save_dir = neox_args.distil_data_gen['save_dir']
#     save_hidden_state = neox_args.distil_data_gen['save_hidden_state']

#     dataset_indicies_path = model_output_filename_path.replace(".dat", ".index")
#     print(f"Iter {iteration}/{neox_args.train_iters} " ,
#             f"Rank {torch.distributed.get_rank()}: "
#             f"Saving model output to {model_output_filename_path}, "
#             f"and dataset index to {dataset_indicies_path}")
#     np_memmap_array.flush() 
#     del np_memmap_array
#     dataset_indicies = np.asarray(dataset_indicies)
#     np.save(dataset_indicies_path, dataset_indicies)

#     torch.distributed.barrier()
#     if torch.distributed.get_rank() == 0:

#         print("Combining memmap array of all ranks")
#         _index_and_dat_files = os.listdir(save_dir)
#         files_to_combine = []
#         for filename in _index_and_dat_files:
#             if "_all" in filename or f"iter{iteration}." not in filename: continue
#             if filename.endswith(".dat") or filename.endswith(".index.npy"):
#                 if f"_iter{iteration}." in filename:
#                     files_to_combine.append(filename)

#         files_to_combine = sorted(files_to_combine)
#         model_output_files = [filename for filename in files_to_combine 
#                                         if filename.endswith(".dat")]

#         all_indicies_files = [np.load(os.path.join(save_dir, filename.replace(".dat", ".index.npy"))) 
#                                         for filename in model_output_files]
#         all_indicies = np.concatenate(all_indicies_files)
#         batch_size = all_indicies.shape[0]

#         (np_memmap_array, 
#         model_output_filename_path) = numpy_memmap_file(neox_args, 
#                                                         iteration, 
#                                                         batch_size = batch_size,
#                                                         start_index=min(all_indicies),
#                                                         end_index=max(all_indicies))

#         sorted_arg_indicies= np.argsort(all_indicies)
#         start = 0
#         for i, filename in enumerate(model_output_files):
#             dataset_indicies = all_indicies_files[i]
#             end = start + dataset_indicies.shape[0]
#             tensor_shape = (dataset_indicies.shape[0], 
#                             neox_args.seq_length, 
#                             neox_args.hidden_size if save_hidden_state else neox_args.padded_vocab_size)
#             saved_model_output_filename_path = os.path.join(save_dir, filename)
#             dtype = 'float16' if neox_args.fp16['enabled'] else 'float32'
#             np_memmap_array_of_rank = np.memmap(saved_model_output_filename_path, dtype=dtype, mode='r+', shape=tensor_shape)
#             np_memmap_array[sorted_arg_indicies[start:end], :, :] = np_memmap_array_of_rank
#             start = end
#             del np_memmap_array_of_rank

#         dataset_indicies_path = model_output_filename_path.replace(".dat", ".index")
#         print(f"Rank all: ",
#             f"Saving model output to {model_output_filename_path} "
#             f"and dataset index to {dataset_indicies_path}")
#         np_memmap_array.flush() 
#         del np_memmap_array
#         # dataset_indicies = np.asarray(dataset_indicies)
#         np.save(dataset_indicies_path, all_indicies[sorted_arg_indicies])

#         for filename in files_to_combine:
#             os.remove(os.path.join(save_dir, filename))
#     torch.distributed.barrier()

# def generate_and_save(
#     neox_args, data_iterator, model, verbose=False
# ):
#     """Evaluation.
#     neox_args: NeoX Arguments
#     forward_step_fn: function with args `neox_args, timers,
#                     data_iterator & model that will run a forward pass on the model
#     data_iterator: Iterator that iterates over batches of data. Should return data in the form:
#                     {'text': np.array([tokens], dtype=np.int64)}
#                     where the size of the array is the model's context size + 1
#                     (`get_batch` transforms it into inputs / labels)
#     """
#     # Turn on evaluation mode which disables dropout.

#     model.eval()
#     save_interval = neox_args.distil_data_gen['save_interval']

#     iteration = 0
#     dataset_indicies = []
#     start_index, end_index = 0, 0
#     (np_memmap_array, 
#     model_output_filename_path) = numpy_memmap_file(neox_args, 
#                                                     iteration, 
#                                                     rank = torch.distributed.get_rank())

#     try:
#         while iteration < neox_args.train_iters:
#             if verbose and iteration % neox_args.log_interval == 0:
#                 print_rank_0(
#                     "Iter {}/{}".format(iteration, neox_args.train_iters)
#                 )

#             dataset_indicies += data_iterator._index_sampler.current_batch_idx

#             with torch.no_grad():
#                 # Forward evaluation
#                 outputs = forward_step(
#                     model=model,
#                     data_iterator=data_iterator,
#                     neox_args=neox_args,
#                 )

#             end_index = start_index + outputs.shape[0]
#             # assert np_memmap_array[start_index: end_index, :, :].shape == outputs.shape, \
#             #         f'Shape mismatch {np_memmap_array[start_index: end_index, :, :].shape} != {outputs.shape}'
#             np_memmap_array[start_index: end_index, :, :] = outputs.cpu().detach().numpy()
#             start_index = end_index
#             # When contiguous memory optimizations are enabled, the buffers
#             # allocated by the optimizations are deallocated during backward pass
#             # in the absence of backward pass the buffers should be reset after each
#             # forward pass
#             if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
#                 deepspeed.checkpointing.reset()

#             if iteration % save_interval == 0:

#                 # TODO save last iter if the iteration<save_interval
#                 save_output(neox_args,
#                             np_memmap_array, 
#                             dataset_indicies, 
#                             model_output_filename_path)
                
#                 start_index, end_index = 0,0 
#                 dataset_indicies = []
#                 (np_memmap_array, 
#                 model_output_filename_path) = numpy_memmap_file(neox_args, 
#                                                                 iteration, 
#                                                                 rank = torch.distributed.get_rank())
#             iteration += 1

#     except Exception as e:       
#         raise e
#     finally: 
#         save_output(neox_args,
#                     np_memmap_array, 
#                     dataset_indicies, 
#                     model_output_filename_path)

#     # Move model back to the train mode.
#     model.train()

# def calulaute_n_iters(neox_args):

#     indexed_dataset = make_indexed_dataset(path=neox_args.data_path,
#                                            impl=neox_args.data_impl)
#     total_num_of_documents = indexed_dataset.sizes.shape[0]   
#     sizes =  indexed_dataset.sizes
#     seq_length = neox_args.seq_length
#     documents = np.arange(start=0, stop=total_num_of_documents,
#                           step=1, dtype=np.int32)
#     tokens_per_epoch = _num_tokens(documents, sizes)
#     return int(((tokens_per_epoch - 1) // seq_length) / neox_args.train_batch_size)

# def get_model():
#     args = {
#         "train_iters" : -1,
#         "eval_interval" : -1,
#         "eval_iters" : -1,
#         "split" : "1,0,0",
#         "gradient_accumulation_steps" : 1,
#         "pipe_parallel_size" : 0,
#         "mmap_warmup" : False,
#         "fp16": { 
#         "enabled": True,
#         }
#     }
#     print_rank_0(f"Overwrite a few of neox args to : {args}")

#     model, neox_args = setup_for_inference_or_eval(inference=False, get_key_value=False, overwrite_values=args)
#     #making the model to retunr hidden state as ouput
#     save_hidden_states = True
#     if save_hidden_states:
#         model.module.activation_checkpoint_interval = 0
#         model.module.sequential = model.module.sequential[:-1]

#     return model, neox_args

# def main():
#     #TODO: make sure model is loaded
#     #save the config of loaded model
#     model, neox_args = get_model()
#     neox_args.train_iters = calulaute_n_iters(neox_args)
#     train_data_iterator, _, _, = build_train_valid_test_data_iterators(neox_args=neox_args)
#     generate_and_save(neox_args, train_data_iterator, model, verbose=False)

# if __name__ == "__main__":
#   main()



