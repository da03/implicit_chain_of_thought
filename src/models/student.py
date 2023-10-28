import sys
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.append("..")
from utils import get_sep_position
from .configuration_student import StudentConfig
from .modeling_gpt2_implicit import GPT2LMHeadImplicitModel

class Student(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_model = GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        num_layers = len(self.base_model.transformer.h)
        hidden_size = self.base_model.config.hidden_size

        self.mlps = nn.ModuleList([nn.Sequential(
                 nn.Linear(hidden_size, 4*hidden_size),
                 nn.ReLU(),
                 nn.Linear(4*hidden_size, hidden_size),
                 ) for _ in range(num_layers)])

    def forward(self, input_ids, positions_to_substitute, teacher_states, output_hidden_states=False):
        outputs = self.base_model.forward(mode='forward_student', \
                input_ids=input_ids, \
                positions_to_substitute=positions_to_substitute, \
                states_to_substitute=teacher_states, \
                output_hidden_states=output_hidden_states)
        return outputs

    def generate(self, input_ids, teacher_states, max_new_tokens=512, num_beams=1):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]
        beam_output = []
        # First, project teacher states
        teacher_states = [self.mlps[l](teacher_states[l]) for l in range(len(teacher_states))]
        for i in range(batch_size):
            input_ids_i = input_ids[i:i+1]
            sep_positions_i = sep_positions[i:i+1]
            input_ids_i = input_ids_i[:sep_positions_i+1]
            beam_output_i = self.base_model.generate(
                input_ids=input_ids_i,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=1,
                positions_to_substitute=sep_positions_i.repeat_interleave(num_beams, dim=0),
                states_to_substitute=[z[i:i+1].repeat_interleave(num_beams, dim=0) for z in teacher_states],
                mode='forward_student',
            )
            beam_output.append(beam_output_i)
        return beam_output

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = StudentConfig.from_pretrained(pretrained_path)
        model = Student(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))

