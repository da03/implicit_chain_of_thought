import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList

from .configuration_teacher import TeacherConfig
import sys
sys.path.append("..")
from utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor
from .modeling_gpt2_implicit import GPT2LMHeadImplicitModel


class Teacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_model = GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        num_layers = len(self.base_model.transformer.h)
        hidden_size = self.base_model.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        outputs = self.base_model.forward(input_ids=input_ids)
        return outputs

    def compute_positions_to_extract_per_layer(self, subset, delta, first_sep_positions, second_sep_positions):
        batch_size = first_sep_positions.shape[0]
        positions_to_extract_per_layer = first_sep_positions.new_zeros(batch_size, self.num_layers).long()
        layer_ids = torch.arange(start=0, end=self.num_layers).to(first_sep_positions.device)
        for batch_id in range(batch_size):
            first_position_to_extract = first_sep_positions[batch_id]
            last_position_to_extract = second_sep_positions[batch_id]
            if subset == 'diagonal':
                if delta == 'dynamic': # determine actual delta
                    delta = (last_position_to_extract - first_position_to_extract) / (self.num_layers - 1)
            elif subset == 'first_column' or subset == 'last_column':
                delta = 0
            else:
                assert subset == 'last_column', subset
                delta = 0
                first_position_to_extract = last_position_to_extract
            positions_to_extract = torch.round(first_position_to_extract + layer_ids * delta)
            positions_to_extract = positions_to_extract.clamp(max=last_position_to_extract)
            positions_to_extract_per_layer[batch_id] = positions_to_extract
        return positions_to_extract_per_layer

    def extract_states(self, input_ids, delta, subset='diagonal'):
        if delta.isnumeric():
            delta = int(delta)
        batch_size = input_ids.shape[0]
        hidden_size = self.hidden_size
        # Forward the teacher to produce all hidden states
        outputs = self.base_model.forward(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[:-1]

        if subset == None:
            return hidden_states

        # Find the boundaries between input and CoT, and CoT and output
        # [input] first_sep_position [CoT] second_position [output] eos
        first_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=0)
        second_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=1)
        input_ids = input_ids[:, :second_sep_positions.max()+1]

        # Compute the positions to extract teacher states (t_l in the paper)
        positions_to_extract_per_layer = self.compute_positions_to_extract_per_layer(subset, delta, first_sep_positions, second_sep_positions)

        # Extract teacher states
        teacher_states_extracted = []
        for i, hidden_state in enumerate(hidden_states):
            if subset == 'diagonal' or subset == 'first_column' or subset == 'last_column':
                z = hidden_state.gather(1, positions_to_extract_per_layer[:,i].view(-1, 1, 1).expand(-1, -1, hidden_size)).squeeze(1)
            elif subset == 'top_row':
                z = hidden_states[-1].gather(1, positions_to_extract_per_layer[:,i].view(-1, 1, 1).expand(-1, -1, hidden_size)).squeeze(1)
            else:
                assert subset == 'bottom_row', subset
                z = hidden_states[0].gather(1, positions_to_extract_per_layer[:,i].view(-1, 1, 1).expand(-1, -1, hidden_size)).squeeze(1)
            # Apply layer norm to normalize to 0 mean and 1 std
            z = self.layer_norm(z)
            teacher_states_extracted.append(z)
        return teacher_states_extracted

    def compute_loss(self, input_ids, labels):
        #import pdb; pdb.set_trace()
        outputs = self.forward(input_ids=input_ids)
        logits = outputs.logits

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        return outputs

    def generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]

        # Since there's one eos after CoT and another after final answer, we need to wait for two eos
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        if stop_on_two_eos:
            generation_config.eos_token_id = -1
            logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
            stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])
        else:
            logits_processor = None
            stopping_criteria = None

        if sep_positions.eq(sep_positions[0]).all():
            input_ids = input_ids[:, :sep_positions[0]+1]
            beam_output = self.base_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=1,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )
            beam_output = beam_output.unsqueeze(1)
        else:
            beam_output = []
            for i in range(batch_size):
                input_ids_i = input_ids[i:i+1]
                sep_positions_i = sep_positions[i:i+1]
                input_ids_i = input_ids_i[:, :sep_positions_i+1]
                beam_output_i = self.base_model.generate(
                    input_ids=input_ids_i,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                    num_return_sequences=1,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )
                beam_output.append(beam_output_i)
        return beam_output

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = TeacherConfig.from_pretrained(pretrained_path)
        model = Teacher(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))
