import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union, Dict, Any

class GPT2ImplicitModel(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        zs=None,
        mult_p=0,
        no_mixture=0,
        softmax_p=0,
        softmax_temperature=1,
        mlps=None,
        relevant_tokens=None,
        mixture_components=None,
        rnn=None,
        key_proj=None,
        query_proj=None,
        out_proj=None,
        attended_to=None,
        attended_to_mask=None,
        positions_to_take=None,
        positions_to_substitute=None,
        states_to_substitute=None,
        mode=None,
        residual=False,
        requires_backward=False,
        phase2=False,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)


        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        zs = []
        f_h_cs = []
        #import pdb; pdb.set_trace()
        if rnn is not None:
            rnn_state = None
        if key_proj is not None:
            assert rnn is not None
            past_keys = None # bsz, len, hidden_size
            context = None
        if mode == 'forward_emulator':
            weight = mixture_components.weight # vocab, hidden_size
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            #assert zs is None
            #assert zs[i] is not None
            hidden_size = hidden_states.shape[-1]
            #import pdb; pdb.set_trace()
            # Gather relevant hidden states at the separator
            if mode == 'forward_emulator':
                z = hidden_states.gather(1, positions_to_take.view(-1, 1, 1).expand(-1, -1, hidden_size)).squeeze(1) # bsz, hidden_size
                #if not phase2:
                #    zs_p.append(zp)
                #else:
                #    zs_q.append(zp)
                zs.append(z)
                c = z # bsz, hidden_size

                #if softmax_p == 0:
                #    with torch.no_grad():
                #        log_probs = c @ weight.T # bsz, vocab
                #        relevant_tokens_i_pred = log_probs.argmax(-1)
                #        if no_mixture == 1:
                #            relevant_tokens_i_pred = relevant_tokens_i_pred * 0
                #        #import pdb; pdb.set_trace()
                #    relevant_proj_pred = mixture_components(relevant_tokens_i_pred) # bsz, hidden_size
                #else:
                #import pdb; pdb.set_trace()
                if weight.shape[0] == 1:
                    mixture_embedding = weight.expand(batch_size, -1)
                else:
                    log_probs = c @ weight.T # bsz, vocab
                    log_probs = log_probs / softmax_temperature
                    probs = log_probs.softmax(dim=-1) # bsz, vocab
                    #relevant_proj_pred = probs @ weight # bsz, H
                    mixture_embedding = probs @ weight # bsz, H
                f_h_c = mlps[i](torch.cat((z, mixture_embedding), dim=-1)) # bsz, hidden_size

                #if phase2:
                #    zs_p.append(f_h_c)
                f_h_cs.append(f_h_c)
                next_input = f_h_c
                if rnn is not None:
                    #import pdb; pdb.set_trace()
                    if key_proj is not None:
                        if context is None:
                            context = next_input.new_zeros(next_input.shape)
                        output, rnn_state = rnn((next_input+context).unsqueeze(0), rnn_state)
                        output = output.squeeze(0)
                        current_key = key_proj(output)
                        if past_keys is not None:
                            current_query = query_proj(output) # bsz, hidden_size
                            attn_weights = torch.bmm(past_keys, current_query.unsqueeze(-1)) # bsz, len, 1
                            attn_probs = attn_weights.softmax(dim=1)
                            attn_probs = attn_probs.squeeze(-1).unsqueeze(1)
                            context = torch.bmm(attn_probs, past_keys).squeeze(1)
                            past_keys = torch.cat((past_keys, current_key.unsqueeze(1)), dim=1)
                        else:
                            past_keys = current_key.unsqueeze(1)
                        output = out_proj(torch.cat((output, context), dim=-1))
                        next_input = output
                    else:
                        rnn_output, rnn_state = rnn(next_input.unsqueeze(0), rnn_state)
                        next_input = rnn_output.squeeze(0)

                #zs_p.append(hidden_states.gather(1, positions_to_take.view(-1, 1, 1).expand(-1, -1, hidden_size)).squeeze(1))
                hidden_states_orig = hidden_states
                if requires_backward:
                    hidden_states = hidden_states.clone()
                if positions_to_take.eq(positions_to_take[0]).all():
                    hidden_states[:, positions_to_take[0]] = next_input
                else:
                    for batch_id in range(positions_to_take.shape[0]):
                        hidden_states[batch_id, positions_to_take[batch_id]] = next_input[batch_id]
            elif mode == 'forward_student':
                assert states_to_substitute is not None
                hidden_size = hidden_states.shape[-1]
                #zs.append(hidden_states.gather(1, first_ids.view(-1, 1, 1).expand(-1, -1, hidden_size)).squeeze(1))
                hidden_states_orig = hidden_states
                if requires_backward:
                    hidden_states = hidden_states.clone()
                if positions_to_substitute.eq(positions_to_substitute[0]).all():
                    hidden_states[:, positions_to_substitute[0]] = states_to_substitute[i]
                else:
                    for batch_id in range(batch_size):
                        hidden_states[batch_id, positions_to_substitut[batch_id]] = states_to_substitute[i][batch_id]


            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        outputs.zs = zs
        outputs.f_h_cs = f_h_cs
        return outputs

class GPT2LMHeadImplicitModel(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2ImplicitModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        mode=None,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        zs = None,
        mult_p=0,
        softmax_p=0,
        no_mixture=0,
        softmax_temperature=1,
        mlps = None,
        rnn=None,
        key_proj=None,
        query_proj=None,
        out_proj=None,
        relevant_tokens=None,
        mixture_components=None,
        positions_to_take=None,
        positions_to_substitute=None,
        states_to_substitute=None,
        residual=False,
        requires_backward=False,
        phase2=False,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #import pdb; pdb.set_trace()
        transformer_outputs = self.transformer.forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            zs=zs,
            mult_p=mult_p,
            softmax_p=softmax_p,
            no_mixture=no_mixture,
            softmax_temperature=softmax_temperature,
            mlps=mlps,
            key_proj=key_proj,
            query_proj=query_proj,
            out_proj=out_proj,
            relevant_tokens=relevant_tokens,
            mixture_components=mixture_components,
            rnn=rnn,
            phase2=phase2,
            positions_to_take=positions_to_take,
            positions_to_substitute=positions_to_substitute,
            states_to_substitute=states_to_substitute,
            residual=residual,
            requires_backward=requires_backward,
            mode=mode,
        )
        zs = transformer_outputs.zs
        f_h_cs = transformer_outputs.f_h_cs
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        outputs = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        outputs.zs = zs
        outputs.f_h_cs = f_h_cs
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, positions_to_substitute=None, states_to_substitute=None, mode=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        if positions_to_substitute is not None:
            model_inputs['positions_to_substitute'] = positions_to_substitute
            model_inputs['states_to_substitute'] = states_to_substitute
            model_inputs['mode'] = mode
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # Remove positions_to_substitute
        if 'positions_to_substitute' in model_kwargs:
            del model_kwargs['positions_to_substitute']
            del model_kwargs['states_to_substitute']
            del model_kwargs['mode']
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs
