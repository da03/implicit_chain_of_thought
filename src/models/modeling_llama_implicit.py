import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Tuple, Union, Dict, Any, List
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

class LlamaImplicitModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        zs = []
        f_h_cs = []
        if rnn is not None:
            rnn_state = None
        if key_proj is not None:
            assert rnn is not None
            past_keys = None # bsz, len, hidden_size
            context = None
        if mode == 'forward_emulator':
            weight = mixture_components.weight # vocab, hidden_size

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_size = hidden_states.shape[-1]
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
                        hidden_states[batch_id, positions_to_substitute[batch_id]] = states_to_substitute[i][batch_id]


            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        outputs.zs = zs
        outputs.f_h_cs = f_h_cs
        return outputs


class ImplicitLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaImplicitModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        mode=None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
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
        zs = outputs.zs
        f_h_cs = outputs.f_h_cs

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        outputs.zs = zs
        outputs.f_h_cs = f_h_cs
        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, positions_to_substitute=None, states_to_substitute=None, mode=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
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
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

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
