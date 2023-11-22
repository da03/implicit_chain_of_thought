import sys
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.bert.modeling_bert import BertModel

sys.path.append("..")
from utils import get_sep_position
from .configuration_autoencoder import AutoEncoderConfig
from .modeling_gpt2_implicit import GPT2LMHeadImplicitModel

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        teacher_num_layers = config.teacher_num_layers
        self.base_model_decoder = GPT2Model.from_pretrained(config.base_model)
        #self.base_model_encoder = nn.ModuleList([BertModel.from_pretrained('bert-base-uncased') for _ in range(teacher_num_layers)]) #GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        self.base_model_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        #import pdb; pdb.set_trace()
        #num_layers = len(self.base_model.transformer.h)
        #hidden_size = self.base_model[0].config.hidden_size
        encoder_hidden_size = self.base_model_encoder.config.hidden_size
        decoder_hidden_size = self.base_model_decoder.config.hidden_size
        teacher_hidden_size = config.teacher_hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.layer_norm = nn.LayerNorm(teacher_hidden_size, elementwise_affine=False)
        #self.num_layers = num_layers
        #self.hidden_size = hidden_size
        self.mlp_in = nn.Linear(teacher_hidden_size*teacher_num_layers, encoder_hidden_size)
        self.mlp_out = nn.Linear(encoder_hidden_size, teacher_hidden_size*teacher_num_layers)
        self.mlp_in_decoder = nn.Linear(teacher_hidden_size*teacher_num_layers, decoder_hidden_size)
        self.mlp_out_decoder = nn.Linear(decoder_hidden_size, teacher_hidden_size*teacher_num_layers)
        #self.mlps_in = nn.ModuleList([nn.Sequential(
        #         nn.Linear(teacher_hidden_size, hidden_size),
        #         ) for _ in range(teacher_num_layers)])
        #self.mlps_out = nn.ModuleList([nn.Sequential(
        #         nn.Linear(hidden_size, teacher_hidden_size),
        #         ) for _ in range(teacher_num_layers)])

    def forward(self, input_ids, positions_to_substitute, teacher_states, output_hidden_states=False):
        outputs = self.base_model.forward(mode='forward_student', \
                input_ids=input_ids, \
                positions_to_substitute=positions_to_substitute, \
                states_to_substitute=teacher_states, \
                output_hidden_states=output_hidden_states)
        return outputs

    def encode(self, teacher_states_cat):
        #import pdb; pdb.set_trace()
        batch_size = teacher_states_cat.shape[0]
        seq_len = teacher_states_cat.shape[1]
        teacher_states_cat = teacher_states_cat.view(batch_size, seq_len, -1)
        state_in = self.mlp_in(teacher_states_cat)
        state_out = self.base_model_encoder(inputs_embeds=state_in).last_hidden_state
        bottleneck = state_out.mean(1)
        bottleneck = self.mlp_out(bottleneck)
        bottleneck = bottleneck.view(batch_size, -1, self.teacher_hidden_size)
        bottleneck = self.layer_norm(bottleneck) # bsz, layers, hidden
        return bottleneck

    def compute_loss(self, teacher_states):
        teacher_states_cat = torch.stack(teacher_states, dim=-2) # bsz, seq_len, layers, hidden
        batch_size = teacher_states_cat.shape[0]
        seq_len = teacher_states_cat.shape[1]
        bottleneck = self.encode(teacher_states_cat)
        teacher_states_cat = teacher_states_cat.view(batch_size, seq_len, -1)
        
        bottleneck = bottleneck.view(batch_size, 1, -1) # bz, 1, layers*hidden
        inputs_embeds = teacher_states_cat + bottleneck # bsz, seq_len, layers*hidden
        inputs_embeds = torch.cat((bottleneck, inputs_embeds), dim=1) # bsz, 1+seq_len, layers*hidden
        inputs_embeds = self.mlp_in_decoder(inputs_embeds) # bsz, 1+seq_len, hidden
        outputs = self.base_model_decoder(inputs_embeds=inputs_embeds)
        last_hidden_state = outputs.last_hidden_state
        outputs = self.mlp_out_decoder(last_hidden_state)
        outputs = outputs[:, :-1]

        loss_fct = nn.MSELoss(reduction='none')
        loss = loss_fct(teacher_states_cat, outputs).sum(-1) / 2
        loss = loss.mean(0).sum(-1)
        outputs = CausalLMOutputWithCrossAttentions(loss=loss)
        outputs.total_loss = loss * batch_size
        # Decoder
        #teacher_states_cat_in = teacher_states_cat + bottleneck.view(
        #encoded_states = []
        #for layer_id, state in enumerate(teacher_states):
        #    state_in = self.mlps_in[layer_id](state)
        #    state_out = self.base_model_encoder[layer_id](inputs_embeds=state_in).last_hidden_state
        #    bottleneck = state_out.mean(1, keepdim=True)
        #    bottleneck = self.layer_norm(bottleneck)
        #    encoded_states.append(bottleneck)
        #encoded_states = torch.cat(encoded_states, 1)
        #sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        ## First, project teacher states
        #teacher_states = [self.mlps[l](teacher_states[l]) for l in range(len(teacher_states))]

        ## Forward while substituting teacher states
        #outputs = self.forward(input_ids, sep_positions, teacher_states)
        #logits = outputs.logits

        #labels_pred = logits.argmax(-1)
        #mask = labels[...,1:].ge(0)
        #correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        #total_tokens = mask.sum()
        #token_accuracy = correct_tokens / total_tokens

        #shift_logits = logits[..., :-1, :].contiguous()
        #shift_labels = labels[..., 1:].contiguous()
        #loss_fct = nn.CrossEntropyLoss()
        #loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        #outputs.loss = loss
        #outputs.token_accuracy = token_accuracy
        #outputs.total_correct = correct_tokens
        #outputs.total_loss = loss * total_tokens
        #outputs.total_tokens = total_tokens
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
        config = AutoEncoderConfig.from_pretrained(pretrained_path)
        model = AutoEncoder(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))

