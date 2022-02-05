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

def forward_step(data_iterator, model, neox_args):
    """Forward step."""

    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        neox_args=neox_args, data_iterator=data_iterator
    )
    
    outputs = model((tokens, position_ids, attention_mask))

    return outputs

def numpy_memmap_file(neox_args, iteration, save_interval, save_to, save_hidden_state=True):
    if not os.path.isdir(save_to):
        os.makedirs(save_to)

    model_output_filename = f"model_output_grank{torch.distributed.get_rank()}_iter{iteration}.dat"
    model_output_filename_path = os.path.join(save_to, model_output_filename)
    if save_hidden_state:
        tensor_shape = (save_interval*neox_args.train_micro_batch_size_per_gpu, 
                        neox_args.seq_length, 
                        neox_args.hidden_size)
    else:
        tensor_shape = (save_interval*neox_args.train_micro_batch_size_per_gpu, 
                        neox_args.seq_length, 
                        neox_args.padded_vocab_size)

    dtype = 'float16' if neox_args.fp16['enabled'] else 'float32'
    fp16_np_memmap_array = np.memmap(model_output_filename_path, dtype=dtype, mode='w+', shape=tensor_shape)

    return fp16_np_memmap_array, model_output_filename_path

def generate_and_save(
    neox_args, data_iterator, model, verbose=False
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
    save_interval = 100

    iteration = 0
    dataset_indicies = []
    start_index, end_index = 0, 0
    (fp16_np_memmap_array, 
    model_output_filename_path) = numpy_memmap_file(neox_args, 
                                                    iteration, 
                                                    save_interval, 
                                                    save_to = "model_output",
                                                    save_hidden_state=True)

    try:
        while iteration < neox_args.train_iters:
            iteration += 1
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Iter {}/{}".format(iteration, neox_args.train_iters)
                )

            dataset_indicies += data_iterator._index_sampler.current_batch_idx

            with torch.no_grad():
                # Forward evaluation
                outputs = forward_step(
                    model=model,
                    data_iterator=data_iterator,
                    neox_args=neox_args,
                )

            end_index = start_index + outputs.shape[0]
            fp16_np_memmap_array[start_index: end_index, :, :] = outputs.cpu().detach().numpy()
            start_index = end_index
            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            if iteration % save_interval == 0:
                fp16_np_memmap_array.flush() 
                dataset_indicies = np.asarray(dataset_indicies)
                np.save(model_output_filename_path.replace(".dat", ".index"), dataset_indicies)
                    
                start_index, end_index = 0, 0
                dataset_indicies = []
                (fp16_np_memmap_array, 
                model_output_filename_path) = numpy_memmap_file(neox_args, 
                                                                iteration, 
                                                                save_interval, 
                                                                save_to = "model_output",
                                                                save_hidden_state=True)

    except Exception as e:
        raise e
    finally: 
        fp16_np_memmap_array.flush() 
        dataset_indicies = np.asarray(dataset_indicies)
        np.save(model_output_filename_path.replace(".dat", ".index"), dataset_indicies)

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

def get_model():
    args = {
        "train_iters" : -1,
        "eval_interval" : -1,
        "eval_iters" : -1,
        "split" : "1,0,0",
        "gradient_accumulation_steps" : 1,
        "pipe_parallel_size" : 0,
        "mmap_warmup" : False,
        "fp16": { 
        "enabled": True,
        }
    }
    print_rank_0(f"Overwrite a few of neox args to : {args}")

    model, neox_args = setup_for_inference_or_eval(inference=False, get_key_value=False, overwrite_values=args)
    #making the model to retunr hidden state as ouput
    save_hidden_states = True
    if save_hidden_states:
        model.module.activation_checkpoint_interval = 0
        model.module.sequential = model.module.sequential[:-1]

    return model, neox_args

def main():
    model, neox_args = get_model()
    neox_args.train_iters = calulaute_n_iters(neox_args)
    train_data_iterator, _, _, = build_train_valid_test_data_iterators(neox_args=neox_args)
    generate_and_save(neox_args, train_data_iterator, model, verbose=False)

if __name__ == "__main__":
  main()
