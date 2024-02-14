import os

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .configuration_emulator import EmulatorConfig
import sys
sys.path.append("..")
from utils import get_sep_position
from .modeling_gpt2_implicit import GPT2LMHeadImplicitModel
import logging

class Emulator(nn.Module):
    def __init__(self, config, nopretrain=False):
        super().__init__()
        self.config = config
        self.base_model = GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        if nopretrain:
            print ('NO PRETRAIN')
            self.base_model.apply(self.base_model._init_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        num_layers = len(self.base_model.transformer.h)
        hidden_size = self.base_model.config.hidden_size

        self.mlps = nn.ModuleList([nn.Sequential(
             nn.Linear(2*hidden_size, 4*hidden_size),
             nn.ReLU(),
             nn.Linear(4*hidden_size, hidden_size),
             ) for _ in range(num_layers)])

        if config.mixture_size == 'vocab_size':
            config.mixture_size = len(self.tokenizer)
        else:
            config.mixture_size = int(config.mixture_size)
        print (f'MIXTURE SIZE: {config.mixture_size}')
        self.mixture_components = nn.Embedding(config.mixture_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, \
                batch_first=False, dropout=0, bidirectional=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size*2, hidden_size)

    def eval(self):
        self.base_model.eval()

    def forward_outputs(self, input_ids, requires_backward=False, supervised_mixture_components=None):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        input_ids = input_ids[:, :sep_positions.max()+1]
        outputs = self.base_model.forward(mode='forward_emulator', \
                input_ids=input_ids, \
                positions_to_take=sep_positions, \
                softmax_temperature=self.config.softmax_temperature, \
                requires_backward=requires_backward, \
                rnn=self.rnn, \
                mlps=self.mlps, \
                mixture_components=self.mixture_components, \
                key_proj=self.key_proj, \
                query_proj=self.query_proj, \
                out_proj=self.out_proj, \
                supervised_mixture_components=supervised_mixture_components)
        return outputs

    def forward(self, input_ids, requires_backward=False, supervised_mixture_components=None):
        outputs = self.forward_outputs(input_ids, requires_backward, supervised_mixture_components)
        emulated_teacher_states = outputs.f_h_cs
        return emulated_teacher_states

    def compute_loss(self, input_ids, teacher_states, supervised_mixture_components=None, mixture_component_supervision_weight=0):
        outputs = self.forward_outputs(input_ids=input_ids, requires_backward=True, \
                supervised_mixture_components=supervised_mixture_components)
        emulated_teacher_states = outputs.f_h_cs
        batch_size = input_ids.shape[0]

        loss_fct = nn.MSELoss(reduction='none')
        loss = 0
        for teacher_state, emulated_teacher_state in zip(teacher_states, emulated_teacher_states):
            loss += loss_fct(teacher_state, emulated_teacher_state).sum(-1) / 2
        loss = loss.mean()

        mixture_component_loss = 0
        if mixture_component_supervision_weight > 0:
            #import pdb; pdb.set_trace()
            mixture_component_log_probs = outputs.mixture_component_log_probs
            for log_probs, mixture_components in zip(mixture_component_log_probs, supervised_mixture_components.T):
                mixture_component_loss += -1.0 * log_probs.gather(1, mixture_components.view(-1, 1)).squeeze(-1)
            mixture_component_loss = mixture_component_loss.mean()

        outputs = CausalLMOutputWithCrossAttentions(loss=loss+mixture_component_loss*mixture_component_supervision_weight)
        outputs.total_loss = loss * batch_size
        outputs.total_mixture_component_loss = mixture_component_loss * batch_size
        return outputs

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = EmulatorConfig.from_pretrained(pretrained_path)
        model = Emulator(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict, strict=False)
            logging.warn("Some weights of the model Emulator checkpoint not loaded.")
        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))
