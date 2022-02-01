# from megatron.model.word_embeddings import EmbeddingPipe, SoftEmbedding
# from megatron.model.transformer import ParallelLinearPipe, NormPipe, ParallelTransformerLayerPipe
# from megatron.model.gmlp import GMLPBlock
from megatron import print_rank_0

class DistilDecorator:

    do_distillation = False

    @staticmethod
    def distil_func(is_class_function=True):
        def inner_func(class_func):
            def inner_inner_func(*args):
                if DistilDecorator.do_distillation:
                    if is_class_function:
                        self, args = args
                        if len(args)!=4:
                            args = args, (None, None, None), (None, None), (None, None)
                        prev_output, input_args, teacher_args, student_args = args

                        # input_args = input_ids, postion_ids, and attention_mask
                        # teacher_args = hidden_states and output_logits
                        # student_args = hidden_states and output_logits

                        class_name = self.__class__.__name__ 

                        # case when input_ids, position_ids, attention_mask, \
                        # and teacher_output_logits are provided as input
                        if class_name in ['EmbeddingPipe', 'SoftEmbedding'] and len(prev_output)==4:
                            input_args, output_logit = prev_output[:3], prev_output[3]
                            prev_output = input_args
                            teacher_args[1] = output_logit

                        # case when input_ids, position_ids, attention_mask, \
                        # and teacher_hidden_states are provided as input
                        if class_name in ['ParallelLinearPipe'] and len(prev_output)==4:
                            input_args, hidden_state = prev_output[:3], prev_output[3]
                            prev_output = hidden_state

                        cur_output = class_func(self, prev_output)

                        # case when input_ids, position_ids, and attention_mask \
                        # are provided as input
                        if class_name in ['EmbeddingPipe', 'SoftEmbedding'] and len(prev_output)==3:
                            if input_args[0] is None: # inside teacher
                                input_args = prev_output
                            else:
                                input_args = (None, None, None)

                        if class_name in ['ParallelLinearPipe'] or \
                            (class_name in ['EmbeddingPipe'] and len(prev_output) not in [3,4]): 
                            hidden_state = prev_output
                            output_logit = cur_output
                            if input_args[0] is not None: # inside teacher
                                teacher_args = (hidden_state, output_logit)
                                cur_output = input_args
                            else:
                                student_args = (hidden_state, output_logit)

                    else:
                        class_name = f"function: {class_func.__name__}"
                        non_class_func = class_func
                        prev_output, input_args, teacher_args, student_args = args[0]
                        cur_output = non_class_func(prev_output)

                    # input_ids, postion_ids, attention_mask = input_args
                    # t_hidden_states, t_output_logits = teacher_args
                    # s_hidden_states, s_output_logits = student_args
                    # print_rank_0(
                    #     class_name,
                    #     "input_ids:", input_ids.shape if input_ids is not None else None,
                    #     "postion_ids:", postion_ids.shape if postion_ids is not None  else None,
                    #     "attention_mask:", attention_mask.shape if attention_mask is not None  else None,
                    #     "t_hidden_states:", t_hidden_states.shape if t_hidden_states is not None  else None,
                    #     "t_output_logits:", t_output_logits.shape if t_output_logits is not None  else None,
                    #     "s_hidden_states:", s_hidden_states.shape if s_hidden_states is not None  else None,
                    #     "s_output_logits:", s_output_logits.shape if s_output_logits is not None  else None,
                    # )

                    return cur_output, input_args, teacher_args, student_args

                else:
                    return class_func(*args)

            return inner_inner_func
        return inner_func

    
# class DistilEmbeddingPipe(EmbeddingPipe):

#     def forward(self, input_args, teacher_args, student_args):
#         input_ids, position_ids, attention_mask = input_args
#         # teacher_args[0] (teacher_hiddden_state) can be non if outputs is already provided
#         is_student = teacher_args[1] is not None
#         hidden_states, outputs = student_args if is_student else teacher_args
#         embeddings, _ = super().forward(input_ids, position_ids)

#         if is_student:
#             student_args = embeddings, outputs
#         else:
#             teacher_args = embeddings, outputs

#         return input_args, teacher_args, student_args

# class DistilSoftEmbedding(SoftEmbedding):

#     def forward(self, input_args, teacher_args, student_args):
#         input_ids, position_ids, attention_mask = input_args
#         is_student = teacher_args[1] is not None
#         hidden_states, outputs = student_args if is_student else teacher_args
#         embeddings, _ = super().forward(input_ids, position_ids)

#         if is_student:
#             student_args = embeddings, outputs
#         else:
#             teacher_args = embeddings, outputs

#         return input_args, teacher_args, student_args


# class DistilParallelTransformerLayerPipe(ParallelTransformerLayerPipe):
    
#     def forward(self, input_args, teacher_args, student_args):

#         input_ids, position_ids, attention_mask = input_args
#         is_student = input_ids is None
#         hidden_states, outputs = student_args if is_student else teacher_args
#         hidden_states, attention_mask = super().forward(hidden_states, attention_mask)

#         if is_student:
#             student_args = hidden_states, outputs
#         else:
#             teacher_args = hidden_states, outputs

#         return input_args, teacher_args, student_args

# class DistilNormPipe(NormPipe):

#     def forward(self, input_args, teacher_args, student_args):

#         input_ids, position_ids, attention_mask = input_args
#         is_student = input_ids is None
#         hidden_states, outputs = student_args if is_student else teacher_args
#         hidden_states = super().forward(hidden_states, attention_mask)

#         if is_student:
#             student_args = hidden_states, outputs
#         else:
#             teacher_args = hidden_states, outputs

#         return input_args, teacher_args, student_args

# class DistilParallelLinearPipe(ParallelLinearPipe):

#     def forward(self, input_args, teacher_args, student_args):
#         input_ids, position_ids, attention_mask = input_args
#         is_student = input_ids is None
#         hidden_states, outputs = student_args if is_student else teacher_args
#         outputs = super().forward(hidden_states, attention_mask)

#         if is_student:
#             student_args = hidden_states, outputs
#         else:
#             teacher_args = hidden_states, outputs

#         return input_args, teacher_args, student_args


# class DistilGMLPBlock(GMLPBlock):

#     def forward(self, input_args, teacher_args, student_args):
#         input_ids, position_ids, attention_mask = input_args
#         is_student = input_ids is None
#         hidden_states, outputs = student_args if is_student else teacher_args
#         hidden_states, attention_mask = super().forward(hidden_states, attention_mask)
        
#         if is_student:
#             student_args = hidden_states, outputs
#         else:
#             teacher_args = hidden_states, outputs

#         return input_args, teacher_args, student_args