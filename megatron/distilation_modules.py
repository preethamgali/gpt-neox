from megatron.model.word_embeddings import EmbeddingPipe, SoftEmbedding
from megatron.model.transformer import ParallelLinearPipe, NormPipe, ParallelTransformerLayerPipe
from megatron.model.gmlp import GMLPBlock

class DistilDecorator:

    do_distillation = False

    @staticmethod
    def distil_func(is_class_function=True):
        def inner_func(class_func):
            def inner_inner_func(*args):
                if DistilDecorator.do_distillation:
                    if is_class_function:
                        self, prev_output, input_args, teacher_args, student_args = args
                        cur_output = class_func(self, prev_output)

                        if self.__class__ in [EmbeddingPipe, SoftEmbedding]:
                            if input_args[0] is None: # inside students
                                input_args = prev_output
                            else:
                                input_args = (None, None, None)

                        if self.__class__ in [ParallelLinearPipe]: 
                            hidden_state = prev_output
                            output = cur_output
                            if input_args[0] is None: # inside teacher
                                student_args = (hidden_state, output)
                                cur_output = input_args
                            else:
                                teacher_args = (hidden_state, output)

                    else:
                        non_class_func = class_func
                        prev_output, input_args, teacher_args, student_args = args
                        cur_output = non_class_func(prev_output)

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