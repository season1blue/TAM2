from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

from transformers.file_utils import ModelOutput

@dataclass
class BaseModelOutput(ModelOutput):
    last_text_state: torch.FloatTensor = None
    last_vision_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    all_generated_vision_hidden_states: torch.FloatTensor = None
    all_generated_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_states: Optional[Tuple[torch.FloatTensor]] = None
    all_cycle_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_cycle_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_patch_policy: Optional[Tuple[torch.FloatTensor]] = None
    all_token_policy: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithPooling(ModelOutput):
    last_text_state: torch.FloatTensor = None
    last_vision_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    all_generated_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_generated_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_states: Optional[Tuple[torch.FloatTensor]] = None
    all_cycle_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_cycle_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_patch_policy: Optional[Tuple[torch.FloatTensor]] = None
    all_token_policy: Optional[Tuple[torch.FloatTensor]] = None

# @dataclass
# class BaseModelOutput(ModelOutput):
#     def __init__(self, last_hidden_state, hidden_states, attentions, all_generated_vision_hidden_states, all_generated_text_hidden_states, vision_states, all_cycle_text_hidden_states, all_cycle_vision_hidden_states, all_patch_policy, all_token_policy)
#     super().__init__()
#     self.last_hidden_state = last
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     all_generated_vision_hidden_states: torch.FloatTensor = None
#     all_generated_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     vision_states: Optional[Tuple[torch.FloatTensor]] = None
#     all_cycle_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     all_cycle_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     all_patch_policy: Optional[Tuple[torch.FloatTensor]] = None
#     all_token_policy: Optional[Tuple[torch.FloatTensor]] = None


# @dataclass
# class BaseModelOutputWithPooling(ModelOutput):
#     """
#     Base class for model's outputs that also contains a pooling of the last hidden states.
#     Args:
#         last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#             Sequence of hidden-states at the output of the last layer of the model.
#         pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
#             Last layer hidden-state of the first token of the sequence (classification token) after further processing
#             through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
#             the classification token after processing through a linear layer and a tanh activation function. The linear
#             layer weights are trained from the next sentence prediction (classification) objective during pretraining.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
#             one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#     """

#     last_hidden_state: torch.FloatTensor = None
#     pooler_output: torch.FloatTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     all_generated_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     all_generated_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     vision_states: Optional[Tuple[torch.FloatTensor]] = None
#     all_cycle_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     all_cycle_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     all_patch_policy: Optional[Tuple[torch.FloatTensor]] = None
#     all_token_policy: Optional[Tuple[torch.FloatTensor]] = None
