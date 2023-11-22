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
        self.num_layers = num_layers
        self.hidden_size = hidden_size

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

    def compute_loss(self, input_ids, labels, teacher_states):
        #import pdb; pdb.set_trace()
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        # First, project teacher states
        teacher_states = [self.mlps[l](teacher_states[l]) for l in range(len(teacher_states))]

        # Forward while substituting teacher states
        outputs = self.forward(input_ids, sep_positions, teacher_states)
        logits = outputs.logits

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
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
            input_ids_i = input_ids_i[:, :sep_positions_i+1]
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
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))

