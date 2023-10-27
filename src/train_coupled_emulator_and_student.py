import math
import re
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import argparse
import os
import inspect
import tqdm
from data import CoTVAEDataset, VAEDataCollator
import logging
import random
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
#torch.autograd.set_detect_anomaly(True)
random.seed(1234)
torch.manual_seed(1234)
logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
# or 
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(key_proj, query_proj, out_proj, model, model_q, mlps, mixture_components,  rnn, mlps_patch, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(mlps.state_dict(), os.path.join(model_dir, 'mlps.pt'))
    torch.save(mixture_components.state_dict(), os.path.join(model_dir, 'mixture_components.pt'))
    if rnn is not None:
        torch.save(rnn.state_dict(), os.path.join(model_dir, 'rnn.pt'))
    if key_proj is not None:
        torch.save(key_proj.state_dict(), os.path.join(model_dir, 'key_proj.pt'))
    if query_proj is not None:
        torch.save(query_proj.state_dict(), os.path.join(model_dir, 'query_proj.pt'))
    if out_proj is not None:
        torch.save(out_proj.state_dict(), os.path.join(model_dir, 'out_proj.pt'))
    torch.save(mlps_patch.state_dict(), os.path.join(model_dir, 'mlps_patch.pt'))
    model_q_dir = os.path.join(model_dir, 'q')
    os.makedirs(model_q_dir, exist_ok=True)
    print ('saving', model_q_dir)
    model_q.save_pretrained(model_q_dir)
    tokenizer.save_pretrained(model_q_dir)

def extract_answer(text):
    ans = text.strip().replace(',', '')
    return ans

def evaluate(no_mixture, key_proj, query_proj, out_proj,softmax_p, softmax_p_temp, mult_p, model, model_q, dataloader, tokenizer, ctx, sigmas, mlps, mixture_components, rnn, mode, mlps_patch, print_eqns=False):
    print_eqns = True
    with torch.no_grad():
        model.eval()
        model_q.eval()
        if print_eqns:
            import collections
            total_finished = 0
            total_unfinished = 0
            total_invalid = 0
            total_valid = 0
            total_eqns = collections.defaultdict(int)
            final_match_eqns = collections.defaultdict(int)
            full_match_eqns = collections.defaultdict(int)
        total = 0
        word_correct = 0
        total_correct = 0
        total_loss = 0
        total_loss_nll = 0
        total_loss_kl = 0
        total_instances = 0
        num_layers = len(model_q.transformer.h)
        for batch in tqdm.tqdm(dataloader):
            input_ids_cot_real = batch['input_ids_cot'].to(device)
            input_ids_cot = batch['input_ids_only'].to(device)
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            labels_cot = batch['labels_cot'].to(device)
            labels_cot_shift = batch['labels_cot_shift'].to(device)
            mask = labels_cot_shift.lt(0)
            labels_nocot = batch['labels_nocot'].to(device)
            with ctx:
                #outputs_cot = model_q(input_ids=input_ids_cot, output_hidden_states=True)
                #hidden_states_cot = outputs_cot.hidden_states

                # now, calculate q: batch_size, hidden_size
                batch_size = input_ids_cot.shape[0]
                #import pdb; pdb.set_trace()
                hidden_size = model_q.config.n_embd
                #hidden_size = hidden_states_cot[0].shape[-1]
                #num_layers = len(hidden_states_cot) - 1
                ###relevant_ids = input_ids_cot.new_zeros(batch_size, num_layers+1).long()
                relevant_ids = input_ids_cot.new_zeros(batch_size, num_layers).long()
                first_ids = input_ids_cot.new_zeros(batch_size).long()
                for batch_id in range(batch_size):
                    mask_id = mask[batch_id]
                    mask_id_list = mask_id.cpu().tolist()
                    first_id = mask_id_list.index(False)
                    first_ids[batch_id] = first_id
                    try:
                        last_id = mask_id_list[first_id:].index(True) + first_id
                    except ValueError:
                        #last_id = len(mask_id_list)
                        last_id = len(mask_id_list)

                    ###layers = torch.arange(start=0, end=num_layers+1)
                    layers = torch.arange(start=0, end=num_layers)
                    ids = torch.round(first_id + 0*layers * (last_id - 1 - first_id) / (num_layers))
                    relevant_ids[batch_id] = ids
                #import pdb; pdb.set_trace()

                # time to compute q
                #hidden_state_relevant_list = []
                #zs0 = []
                #for i, hidden_states in enumerate(hidden_states_cot[:-1]):
                #    hidden_state_relevant = hidden_states.gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1)
                #    hidden_state_relevant = (mlps[i](hidden_state_relevant))
                #    zs0.append(hidden_state_relevant)
                #    #hidden_state_relevant_list.append(hidden_state_relevant + torch.randn_like(hidden_state_relevant) * sigmas[i])
                #    hidden_state_relevant_list.append(hidden_state_relevant)
                #zs = hidden_state_relevant_list


                #import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()
                #outputs_cot = model_q.forward_zs(input_ids=input_ids_cot, zs=None, mlps=mlps, first_ids=first_ids)
                relevant_tokens = input_ids_cot_real.gather(1, relevant_ids)
                model_q.tokenizer = tokenizer
                model_q.transformer.tokenizer = tokenizer
                #outputs_cot = model_q.forward_zs_feedp_pred_predicttoken(input_ids=input_ids_cot, zs=None, first_ids=first_ids, rnn=rnn, mlps=mlps, relevant_tokens=relevant_tokens, mixture_components=mixture_components, phase2=True, mult_p=mult_p, softmax_p=softmax_p, softmax_p_temp=softmax_p_temp)
                outputs_cot = model_q.forward_zs_feedp_pred_predicttoken(input_ids=input_ids_cot, zs=None, first_ids=first_ids, rnn=rnn, mlps=mlps, relevant_tokens=relevant_tokens, mixture_components=mixture_components, phase2=True, mult_p=mult_p, softmax_p=softmax_p, softmax_p_temp=softmax_p_temp, key_proj=key_proj, query_proj=query_proj, out_proj=out_proj, no_mixture=no_mixture)
                hidden_state_relevant_list = outputs_cot.zs_p
                zs = []
                for i, z in enumerate(hidden_state_relevant_list):
                    zs.append(mlps_patch[i](z))
                hidden_state_relevant_list = zs
                outputs_nocot = model.forward_zs(input_ids=input_ids_nocot, zs=hidden_state_relevant_list, first_ids=first_ids)
                #outputs_nocot = model.forward_zs_attn(input_ids=input_ids_nocot, attended_to=hidden_states_cot, attended_to_mask=~mask, first_ids=first_ids, sigmas=sigmas)
                #zs_p = outputs_nocot.zs_p
                #zs_q = zs0 #outputs_nocot.zs_q
            logits = outputs_nocot.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_nocot[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            kl = 0.
            #for z, zp, sigma_i in zip(zs_q, zs_p, sigmas):
            #    kl += ((z-mlp(zp))*(z-mlp(zp))).sum() / sigma_i / sigma_i / 2 / labels_nocot[..., 1:].ge(0).sum().item()
            total_loss += (loss + kl).item() * labels_nocot[...,1:].ge(0).sum().item()
            total_loss_kl += kl * labels_nocot[...,1:].ge(0).sum().item()
            total_loss_nll += loss.item() * labels_nocot[...,1:].ge(0).sum().item()
            if print_eqns:
                weight = mixture_components.weight # vocab, hidden_size
                eqns = [[] for _ in range(batch_size)]
                #outputs_cot = model_q.forward_zs_feedp_pred_predicttoken(input_ids=input_ids_cot, zs=None, first_ids=first_ids, rnn=rnn, mlps=mlps, relevant_tokens=relevant_tokens, mixture_components=mixture_components, phase2=False, mult_p=mult_p, softmax_p=softmax_p, softmax_p_temp=softmax_p_temp, key_proj=key_proj, query_proj=query_proj, out_proj=out_proj)
                hidden_state_relevant_list = outputs_cot.zs_q
                for c in hidden_state_relevant_list:
                    log_probs = c @ weight.T # bsz, vocab
                    log_probs = log_probs.log_softmax(dim=-1)
                    relevant_tokens_i_pred = log_probs.argmax(-1)
                    if print_eqns:
                        for bid in range(batch_size):
                            eqns[bid].append(relevant_tokens_i_pred[bid].item())


            labels_pred = logits.argmax(-1)
            correct = ((labels_pred[...,:-1] == labels_nocot[..., 1:]) * labels_nocot[..., 1:].ge(0)).sum().item()
            word_correct += correct
            total += labels_nocot[..., 1:].ge(0).sum().item()


            # TODO: generate and evaluate accuracy
            # activate beam search and early_stopping
            #import pdb; pdb.set_trace()
            no_brackets = False
            no_brackets = True
            def is_finished(eqn):
                return eqn[-1] == 50256
            def is_valid(eqn):
                es = eqn.split(' ')
                for e in es:
                    if no_brackets:
                        m = re.match(r'.*?=.*?', e)
                    else:
                        m = re.match(r'<<.*?=.*?>>', e)
                    if not m:
                        return False
                return True

            def extract_final_ans(s):
                if no_brackets:
                    m = re.match(r'.*=(.*)', s)
                else:
                    m = re.match(r'.*=(.*)>>', s)
                if not m:
                    return None
                return m.group(1)

            for i, input_ids_single in enumerate(input_ids_nocot):
                sep_id = input_ids_single.tolist().index(tokenizer.eos_token_id)
                #sep_ids.append(sep_id)
            #assert all([item == sep_id for item in sep_ids])
            #import pdb; pdb.set_trace()
                tgt = input_ids_single[sep_id+1:]
                max_new_tokens = tgt.size(0)+10
                max_new_tokens = tgt[tgt.ne(tokenizer.eos_token_id)].size(0)+10
                beam_size = 5
                beam_output = model.generate(
                    input_ids=input_ids_nocot[i, :sep_id+1].unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    num_beams=beam_size,
                    early_stopping=True,
                    num_return_sequences=1,
                    first_ids=first_ids[i:i+1].repeat_interleave(beam_size, dim=0),
                    zs=[z[i:i+1].repeat_interleave(beam_size, dim=0) for z in zs],
                )
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = tgt_text.strip()
                pred_text = tokenizer.decode(beam_output[0][sep_id+1:], skip_special_tokens=True)
                if i == 0:
                    print ("\n" + 100 * '-')
                    print ('GT:', tokenizer.decode(input_ids_single, skip_special_tokens=True))
                    print ('Predicted:', pred_text)
                pred_ans = pred_text.strip() #.split()[-1] #extract_answer(pred_text)
                #import pdb; pdb.set_trace()
                total_instances += 1
                if ans == pred_ans:
                    total_correct += 1
                #i += 1

            #for i, input_ids_single in enumerate(input_ids_nocot):
            #    total_instances += 1
        ##TODO        sep_idx = input_ids_single.tolist().index(tokenizer.eos_token_id)
        ##TODO        src = input_ids_single[:sep_idx+1]
        ##TODO        tgt = input_ids_single[sep_idx+1:]

        ##TODO        sep_idx = tgt.tolist().index(tokenizer.eos_token_id)
        ##TODO        tgt = tgt[:sep_idx]
        ##TODO        tgt_text = tokenizer.decode(tgt)
        ##TODO        ans = extract_answer(tgt_text)
                if print_eqns:
                    #import pdb; pdb.set_trace()
                    eqn = tokenizer.decode(eqns[i], skip_special_tokens=True).strip()
                    sep_idx = input_ids_cot_real[i].tolist().index(tokenizer.eos_token_id)
                    src = input_ids_cot_real[i][:sep_idx+1]
                    src_text = tokenizer.decode(src)
                    tgt = input_ids_cot_real[i][sep_idx+1:]

                    sep_idx = tgt.tolist().index(tokenizer.eos_token_id)
                    tgt = tgt[:sep_idx]
                    tgt_text = tokenizer.decode(tgt)
                    ans = extract_answer(tgt_text)
                    print (f'pred: {eqn}')
                    print (f'gt:   {ans}')
                    if not is_valid(ans):
                        continue
                    ans_es = ans.split(' ')
                    num_es_ans = len(ans_es)
                    if not is_finished(eqns[i]):
                        total_unfinished += 1
                    else:
                        total_finished += 1
                        if not is_valid(eqn):
                            total_invalid += 1
                        else:
                            total_valid += 1
                            total_eqns[num_es_ans] += 1
                            es = eqn.split(' ')
                            final_ans_gt = extract_final_ans(ans_es[-1])
                            final_ans_pred = extract_final_ans(es[-1])
                            if final_ans_gt == final_ans_pred:
                                final_match_eqns[num_es_ans] += 1
                                if len(es) == len(ans_es):
                                    flag = True
                                    for e, ans_e in zip(es, ans_es):
                                        if e != ans_e:
                                            flag = False
                                            break
                                    if flag:
                                        full_match_eqns[num_es_ans] += 1

        ##TODO        with torch.no_grad():
        ##TODO            with ctx:
        ##TODO                beam_size = 5
        ##TODO                #import pdb; pdb.set_trace()
        ##TODO                beam_output = model.generate(
        ##TODO                    input_ids=src.view(1, -1),
        ##TODO                    max_new_tokens=100,
        ##TODO                    num_beams=beam_size,
        ##TODO                    early_stopping=True,
        ##TODO                    num_return_sequences=1,
        ##TODO                    first_ids=first_ids[i:(i+1)].expand(beam_size),
        ##TODO                    zs=[z[i:(i+1)].expand(beam_size, -1) for z in zs],
        ##TODO                )
        ##TODO       
        ##TODO                #import pdb; pdb.set_trace()
        ##TODO                sep_idx = input_ids_single.tolist().index(tokenizer.eos_token_id)
        ##TODO                pred_text = tokenizer.decode(beam_output[0][sep_idx+1:], skip_special_tokens=True)
        ##TODO        if i == 0:
        ##TODO            print ("Output:\n" + 100 * '-')
        ##TODO            print (pred_text)
        ##TODO            sys.stdout.flush()
        ##TODO        pred_ans = extract_answer(pred_text)
        ##TODO        #import pdb; pdb.set_trace()
        ##TODO        if ans == pred_ans:
        ##TODO            total_correct += 1
        ##TODO    #break
        if print_eqns:
            #import pdb; pdb.set_trace()
            if total_finished + total_unfinished > 0:
                print (f'finished: {total_finished / (total_finished + total_unfinished)} ({total_finished} out of {total_finished + total_unfinished})')
                if total_finished > 0:
                    print (f'valid: {total_valid / (total_valid + total_invalid)} ({total_valid} out of {total_valid + total_invalid})')
                    theo_correct = 0
                    for num_digit in final_match_eqns:
                        theo_correct += final_match_eqns[num_digit]
                    print (f'theoretically correct: {theo_correct / (total_instances)} ({theo_correct} out of {total_instances})')

                    for num_digit in [1, 2, 3, 4]:
                        if total_eqns[num_digit] == 0:
                            continue
                        print (f'digit: {num_digit} {total_eqns[num_digit] / total_valid} ({total_eqns[num_digit]} out of {total_valid})')
                        print (f'  - ans match: {final_match_eqns[num_digit] / total_eqns[num_digit]} ({final_match_eqns[num_digit]} out of {total_eqns[num_digit]})')
                        print (f'  - full match: {full_match_eqns[num_digit] / max(final_match_eqns[num_digit], 1)} ({full_match_eqns[num_digit]} out of {final_match_eqns[num_digit]})')

        word_accuracy = word_correct / total
        accuracy = total_correct / total_instances
        loss = total_loss / total
        try:
            ppl = math.exp(loss)
        except Exception as e:
            ppl = float('inf')
        loss_nll = total_loss_nll / total
        ppl_nll = math.exp(loss_nll)
        loss_kl = total_loss_kl / total
    return ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/math_scaffolding_formula/src1_train.txt')
    parser.add_argument('--val_path', type=str, default='data/math_scaffolding_formula/src1_valid.txt')
    parser.add_argument('--test_path', type=str, default='data/math_scaffolding_formula/src1_test.txt')
    parser.add_argument('--eval_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--kl_mean_weight', type=float, default=0.00)
    parser.add_argument('--p_mean_weight', type=float, default=0.00)
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--save_model', type=str, default='model_nocot')
    parser.add_argument('--qmodel', type=str, default='gpt2')
    parser.add_argument('--residual', type=int, default=0)
    parser.add_argument('--additional_norm', type=int, default=0)
    parser.add_argument('--mult_p', type=int, default=0)
    parser.add_argument('--softmax_p', type=int, default=0)
    parser.add_argument('--softmax_p_temp', type=float, default=1.0)
    parser.add_argument('--mode', type=str, choices=['top', 'interleave', 'bottom', 'none'], default='none')
    parser.add_argument('--compile', type=int, default=1)
    parser.add_argument('--no_mixture', type=int, default=0)
    parser.add_argument('--fix_q', type=int, default=0)
    parser.add_argument('--load_model', type=int, default=0)
    parser.add_argument('--fix_p', type=int, default=0)
    parser.add_argument('--follow', type=str, choices=['diagonal_orig', 'diagonal', 'last_column', 'top_row', 'bottom_row', 'bottom_row_above',  'first_column'], default='diagonal')
    parser.add_argument('--interval', type=int, default=-1)
    parser.add_argument('--mixture_size', type=int, default=32)
    args = parser.parse_args()

    print (args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    #if 'gpt2-xl' not in args.model:
    dtype = 'float32'
    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    print (ptdtype, dtype)
    model_q = AutoModelForCausalLM.from_pretrained(args.qmodel).to(device).to(ptdtype)
    print (args.qmodel)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).to(ptdtype)
    if args.compile == 1:
        model = torch.compile(model)
    else:
        print ('WARNING: no compile!')
    #model_q = torch.compile(model_q)
    # TODO: maybe use pretrained model here?
    #model.apply(model._init_weights)
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available
    extra_args = dict(fused=True) if use_fused else dict()
    num_layers = len(model.transformer.h)
    sigmas = torch.ones(num_layers).to(ptdtype).to(device)
    #import pdb; pdb.set_trace()
    hidden_size_in = model.config.hidden_size
    hidden_size_out = model_q.config.hidden_size
    hidden_size_mid = 4 * max(hidden_size_in, hidden_size_out)
    if args.mixture_size == 1:
        a = 1
    else:
        a = args.mixture_size+1
    a = 1
    mlps = nn.ModuleList([nn.Sequential(
             nn.Linear(2*hidden_size_in, hidden_size_mid),
             nn.ReLU(),
             nn.Linear(hidden_size_mid, hidden_size_out * a),
             ) for _ in range(num_layers)]).to(device).to(ptdtype)
    mixture_components = nn.Embedding(len(tokenizer), hidden_size_out).to(device).to(ptdtype)
    if args.load_model == 1:
        mixture_components.load_state_dict(torch.load(os.path.join(args.model, 'mixture_components.pt')))
        mlps.load_state_dict(torch.load(os.path.join(args.model, 'mlps.pt')))
    else:
        mixture_components.load_state_dict(torch.load(os.path.join(args.qmodel, 'mixture_components.pt')))
        mlps.load_state_dict(torch.load(os.path.join(args.qmodel, 'mlps.pt')))
    #mlps_patch = nn.ModuleList([nn.Sequential(
    #         nn.Linear(hidden_size_in, hidden_size_mid),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size_mid, hidden_size_out),
    #         ) for _ in range(num_layers)]).to(device).to(ptdtype)
    if args.load_model == 0:
        if not os.path.exists(os.path.join(args.qmodel, 'rnn.pt')):
            rnn = None
        else:
            rnn = nn.LSTM(input_size=hidden_size_in, hidden_size=hidden_size_in, num_layers=1, batch_first=False, dropout=0, bidirectional=False).to(device).to(ptdtype)
            rnn.load_state_dict(torch.load(os.path.join(args.qmodel, 'rnn.pt')))
    else:
        if not os.path.exists(os.path.join(args.model, 'rnn.pt')):
            rnn = None
        else:
            rnn = nn.LSTM(input_size=hidden_size_in, hidden_size=hidden_size_in, num_layers=1, batch_first=False, dropout=0, bidirectional=False).to(device).to(ptdtype)
            rnn.load_state_dict(torch.load(os.path.join(args.model, 'rnn.pt')))
    if args.load_model == 0:
        if os.path.exists(os.path.join(args.qmodel, 'key_proj.pt')):
            key_proj = nn.Linear(hidden_size_in, hidden_size_out).to(device).to(ptdtype)
            query_proj = nn.Linear(hidden_size_in, hidden_size_out).to(device).to(ptdtype)
            out_proj = nn.Linear(hidden_size_out*2, hidden_size_out).to(device).to(ptdtype)
            key_proj.load_state_dict(torch.load(os.path.join(args.qmodel, 'key_proj.pt')))
            query_proj.load_state_dict(torch.load(os.path.join(args.qmodel, 'query_proj.pt')))
            out_proj.load_state_dict(torch.load(os.path.join(args.qmodel, 'out_proj.pt')))
        else:
            print ('no key, query, out!')
            key_proj, query_proj, out_proj = None, None, None
    else:
        if os.path.exists(os.path.join(args.model, 'key_proj.pt')):
            key_proj = nn.Linear(hidden_size_in, hidden_size_out).to(device).to(ptdtype)
            query_proj = nn.Linear(hidden_size_in, hidden_size_out).to(device).to(ptdtype)
            out_proj = nn.Linear(hidden_size_out*2, hidden_size_out).to(device).to(ptdtype)
            key_proj.load_state_dict(torch.load(os.path.join(args.model, 'key_proj.pt')))
            query_proj.load_state_dict(torch.load(os.path.join(args.model, 'query_proj.pt')))
            out_proj.load_state_dict(torch.load(os.path.join(args.model, 'out_proj.pt')))
        else:
            print ('no key, query, out!')
            key_proj, query_proj, out_proj = None, None, None
    if args.additional_norm == 0:
        mlps_patch = nn.ModuleList([nn.Sequential(
                 nn.LayerNorm(hidden_size_out, elementwise_affine=False),
                 nn.Linear(hidden_size_in, hidden_size_mid),
                 nn.ReLU(),
                 nn.Linear(hidden_size_mid, hidden_size_out),
                 ) for _ in range(num_layers)]).to(device).to(ptdtype)
        if args.load_model == 1:
            mlps_patch = nn.ModuleList([item[1:] for item in mlps_patch])
            mlps_patch.load_state_dict(torch.load(os.path.join(args.model, 'mlps_patch.pt')))
        else:
            mlps_patch.load_state_dict(torch.load(os.path.join(args.model, 'mlps.pt')))
            mlps_patch = nn.ModuleList([item[1:] for item in mlps_patch])
    else:
        assert False
        print ('additional norm')
        mlps_patch = nn.ModuleList([nn.Sequential(
                 nn.LayerNorm(hidden_size_out, elementwise_affine=False),
                 nn.Linear(hidden_size_in, hidden_size_mid),
                 nn.ReLU(),
                 nn.Linear(hidden_size_mid, hidden_size_out),
                 ) for _ in range(num_layers)]).to(device).to(ptdtype)
        mlps_patch.load_state_dict(torch.load(os.path.join(args.model, 'mlps.pt')))
    #mlps_patch.load_state_dict(torch.load(os.path.join(args.model, 'mlps.pt')))
    #sigmas = torch.zeros(num_layers).to(ptdtype).to(device)
    sigmas = torch.nn.Parameter(sigmas)
    #import pdb; pdb.set_trace()
    #optimizer = torch.optim.AdamW([sigmas] + list(model.parameters())+list(model_q.parameters()), lr=args.lr)
    #for p in model_q.parameters():
    #    p.requires_grad = False
    sigmas.requires_grad = False
    #optimizer = torch.optim.AdamW([sigmas] + list(model.parameters()) + list(mlp.parameters()), lr=args.lr)
    #optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlp.parameters()), lr=args.lr)
    #optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlps.parameters()) + list(model_q.parameters()), lr=args.lr)
    use_fused = True 
    extra_args = dict(fused=True) if use_fused else dict()
    if args.fix_q == 1:
        print ('FIXING Q')
        all_params = list(model.parameters()) + list(mlps_patch.parameters())
        for module in [mlps, rnn, model_q, mixture_components, key_proj, query_proj, out_proj]:
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
    elif args.fix_p == 1:
        print ('FIXING P')
        #all_params = list(model.parameters()) + list(mlps_patch.parameters())
        all_params = list(mlps.parameters()) + list(model_q.parameters()) + list(mixture_components.parameters())
        if rnn is not None:
            all_params.extend(list(rnn.parameters()))
        if key_proj is not None:
            all_params.extend(list(key_proj.parameters()))
            all_params.extend(list(query_proj.parameters()))
            all_params.extend(list(out_proj.parameters()))
        for module in [model, mlps_patch]:
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
    else:
        if rnn is not None:
            all_params = list(model.parameters()) + list(mlps.parameters()) + list(rnn.parameters()) + list(model_q.parameters()) + list(mlps_patch.parameters()) + list(mixture_components.parameters())
        else:
            all_params = list(model.parameters()) + list(mlps.parameters()) + list(model_q.parameters()) + list(mlps_patch.parameters()) + list(mixture_components.parameters())
        print ('UNFIX Q')
        #all_params = list(model.parameters()) + list(mlps_patch.parameters())
        if key_proj is not None:
            all_params.extend(list(key_proj.parameters()))
            all_params.extend(list(query_proj.parameters()))
            all_params.extend(list(out_proj.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, **extra_args)
    #optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlps.parameters()) + list(model_q.parameters()), lr=args.lr, **extra_args)
    #optimizer = torch.optim.AdamW([sigmas] + list(model.parameters())+list(model_q.parameters()), lr=args.lr)
    #optimizer_sigmas = torch.optim.SGD([sigmas], lr=args.lr)
    #optimizer = torch.optim.SGD([sigmas] + list(model.parameters())+list(model_q.parameters()), lr=args.lr)

    collate_fn = VAEDataCollator(tokenizer)
    train_dataset = CoTVAEDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTVAEDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_dataset = CoTVAEDataset(tokenizer, args.test_path, 1024)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    if args.eval_path is not None:
        eval_dataset = CoTVAEDataset(tokenizer, args.eval_path, 1024)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    else:
        eval_dataloader = None

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    #compile = True # use PyTorch 2.0 to compile the model to be faster
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)


    #accuracy, word_accuracy, ppl = evaluate(model, model_q, val_dataloader, tokenizer, ctx, sigmas)
    #print (f'Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
    model.eval()
    model_q.eval()
    ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(args.no_mixture, key_proj, query_proj, out_proj, args.softmax_p, args.softmax_p_temp, args.mult_p, model, model_q, val_dataloader, tokenizer, ctx, sigmas, mlps, mixture_components, rnn, args.mode, mlps_patch)
    print (f"Val. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")

    ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(args.no_mixture, key_proj, query_proj, out_proj, args.softmax_p, args.softmax_p_temp, args.mult_p, model, model_q, test_dataloader, tokenizer, ctx, sigmas, mlps, mixture_components, rnn, args.mode, mlps_patch)
    print (f"Test. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Test Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")

    if eval_dataloader is not None:
        ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(args.no_mixture, key_proj, query_proj, out_proj, args.softmax_p, args.softmax_p_temp, args.mult_p, model, model_q, eval_dataloader, tokenizer, ctx, sigmas, mlps, mixture_components, rnn, args.mode, mlps_patch)
        print (f"Eval PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Eval Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")
    #model.train()
    #model_q.train()

    #model.eval()
    #model_q.eval()
    step = 0
    #import pdb; pdb.set_trace()
    for epoch in range(args.epochs):
        save_model(key_proj, query_proj, out_proj, model, model_q, mlps, mixture_components, rnn, mlps_patch, tokenizer, f'{args.save_model}/checkpoint_{epoch}_{args.lr}')
        print(f"Epoch {epoch}") #TODO change epoch

        #model.save_pretrained("finetuned_gpt2")
        for batch in tqdm.tqdm(train_dataloader):
            #import pdb; pdb.set_trace()
            input_ids_cot = batch['input_ids_only'].to(device)
            input_ids_cot_real = batch['input_ids_cot'].to(device)
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            #labels_cot = batch['labels_cot'].to(device)
            labels_cot_shift = batch['labels_cot_shift'].to(device)
            mask = labels_cot_shift.lt(0)
            labels_nocot = batch['labels_nocot'].to(device)
            with ctx:
                #outputs_cot_orig = model_q(input_ids=input_ids_cot, output_hidden_states=True)

                # now, calculate q: batch_size, hidden_size
                batch_size = input_ids_cot.shape[0]
                #hidden_size = hidden_states_cot[0].shape[-1]
                hidden_size = model_q.config.n_embd
                #num_layers = len(hidden_states_cot) - 1
                ###relevant_ids = input_ids_cot.new_zeros(batch_size, num_layers+1).long()
                relevant_ids = input_ids_cot.new_zeros(batch_size, num_layers).long()
                relevant_ids_real = input_ids_cot.new_zeros(batch_size, num_layers).long()
                first_ids = input_ids_cot.new_zeros(batch_size).long()
                for batch_id in range(batch_size):
                    mask_id = mask[batch_id]
                    mask_id_list = mask_id.cpu().tolist()
                    first_id = mask_id_list.index(False)
                    first_ids[batch_id] = first_id
                    try:
                        last_id = mask_id_list[first_id:].index(True) + first_id
                    except ValueError:
                        last_id = len(mask_id_list)

                    ##layers = torch.arange(start=0, end=num_layers+1)
                    layers = torch.arange(start=0, end=num_layers)
                    a = first_id
                    if args.follow == 'diagonal' or args.follow == 'top_row' or args.follow == 'bottom_row' or args.follow == 'bottom_row_above':
                        interval = (last_id - 1 - first_id) / (num_layers-1)
                    elif args.follow == 'diagonal_orig': # todo: rerun experiments with new setting, but don't think this would change things much
                        interval = (last_id - 1 - first_id) / (num_layers)
                    elif args.follow == 'first_column':
                        interval = 0
                    elif args.follow == 'last_column':
                        interval = 0
                        a = last_id - 1
                    else:
                        assert False
                    if args.interval > 0:
                        interval = args.interval
                    ids_real = torch.round(a + layers * interval)
                    ids = torch.round(a + 0*layers * interval)
                    if args.interval > 0:
                        ids_real = ids_real.clamp(max=last_id - 1)
                    #ids = torch.round(first_id + 0*layers * (last_id - 1 - first_id) / (num_layers))
                    assert args.follow == 'diagonal'
                    #assert args.interval == 1
                    #ids_real = torch.round(first_id + layers * (last_id - 1 - first_id) / (num_layers))
                    relevant_ids[batch_id] = ids
                    relevant_ids_real[batch_id] = ids_real
                #import pdb; pdb.set_trace()
                relevant_tokens = input_ids_cot_real.gather(1, relevant_ids_real)
                #outputs_cot = model_q.forward_zs_feedp(input_ids=input_ids_cot, zs=hidden_state_relevant_list, first_ids=first_ids)
                #hidden_states_cot = outputs_cot.hidden_states

                ## time to compute q
                #hidden_state_relevant_list = []
                #zs0 = []
                #for i, hidden_states in enumerate(hidden_states_cot[:-1]):
                #    hidden_state_relevant = hidden_states.gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1)
                #    zs0.append(hidden_state_relevant)
                #    #hidden_state_relevant_list.append(hidden_state_relevant + torch.randn_like(hidden_state_relevant) * sigmas[i])
                #    hidden_state_relevant_list.append(hidden_state_relevant)

                ###for hidden_states in hidden_states_cot:
                ###    hidden_states[mask] = 0 # batch_size, seq_len, hidden_size
                #outputs_cot = model_q.forward_zs(input_ids=input_ids_cot, zs=None, mlps=mlps, first_ids=first_ids, clone=True)
                #outputs_cot = model_q.forward_zs_feedp_pred_predicttoken(input_ids=input_ids_cot, zs=None, first_ids=first_ids, clone=True, rnn=rnn, mlps=mlps, relevant_tokens=None, mixture_components=mixture_components, phase2=True, mult_p=args.mult_p, softmax_p=args.softmax_p, softmax_p_temp=args.softmax_p_temp)
                outputs_cot = model_q.forward_zs_feedp_pred_predicttoken(input_ids=input_ids_cot, zs=None, first_ids=first_ids, clone=True, rnn=rnn, mlps=mlps, relevant_tokens=None, mixture_components=mixture_components, phase2=True, mult_p=args.mult_p, softmax_p=args.softmax_p, softmax_p_temp=args.softmax_p_temp, key_proj=key_proj, query_proj=query_proj, out_proj=out_proj, no_mixture=args.no_mixture)
                extra = 0
                if args.p_mean_weight > 0:
                    weight = mixture_components.weight # vocab, hidden_size
                    for kl_i, z in enumerate(outputs_cot.zs_q):
                        relevant_tokens_i = relevant_tokens[:, kl_i] # bsz
                        c = z # bsz, hidden_size
                        log_probs = c @ weight.T # bsz, vocab
                        log_probs = log_probs.log_softmax(-1) # bsz, vocab

                        kl_ids = relevant_tokens_i
                        log_probs_selected = log_probs.gather(1, kl_ids.view(-1, 1))
                        if step %100 == 0:
                            print (f'rank of i: {kl_i}')
                            log_probs_sorted, ids_sorted_pred = log_probs.sort(dim=-1, descending=True)
                            print(ids_sorted_pred.eq(kl_ids.view(-1, 1)).float().argmax(-1))
                            print (log_probs_selected.exp().view(-1))
                        if args.p_mean_weight > 0:
                            extra += -args.p_mean_weight * log_probs_selected.mean()
                hidden_state_relevant_list = outputs_cot.zs_p
                zs = []
                for i, z in enumerate(hidden_state_relevant_list):
                    zs.append(mlps_patch[i](z))
                hidden_state_relevant_list = zs

                outputs_nocot = model.forward_zs(input_ids=input_ids_nocot, zs=hidden_state_relevant_list, first_ids=first_ids)#, clone=True)
                #outputs_nocot = model.forward_zs_attn(input_ids=input_ids_nocot, attended_to=hidden_states_cot, attended_to_mask=~mask, first_ids=first_ids, sigmas=sigmas)
                #zs_p = outputs_nocot.zs_p
                #zs_q = zs0 #outputs_nocot.zs_q
            #loss = outputs.loss
            logits = outputs_nocot.logits

            labels_pred = logits.argmax(-1)
            #import pdb; pdb.set_trace()
            correct = ((labels_pred[...,:-1] == labels_nocot[...,1:]) * labels_nocot[...,1:].ge(0)).sum().item()
            total = labels_nocot[...,1:].ge(0).sum()
            accuracy = correct / total

            kl = 0.
            #import pdb; pdb.set_trace()
            #for z, zp, sigma_i in zip(zs_q, zs_p, sigmas):
            #    #kl += ((z-zp)*(z-zp)).sum() / sigma_i / sigma_i / 2 / total
            #    kl += ((z-mlp(zp))*(z-mlp(zp))).sum() / sigma_i / sigma_i / 2 / total


            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_nocot[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) 
            loss = nll + kl
            (nll+extra).div(args.accumulate).backward()
            #loss.div(args.accumulate).backward()
            #kl.div(args.accumulate).backward()
            #import pdb; pdb.set_trace()

            if step % args.accumulate == args.accumulate-1:
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(model_q.parameters(), args.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(mlps.parameters(), args.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(rnn.parameters(), args.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(mlps_patch.parameters(), args.max_grad_norm)
                if args.p_mean_weight > 0:
                    all_modules = [model, model_q, mlps_patch, mlps, mixture_components]
                    if key_proj is not None:
                        all_modules.extend([key_proj, query_proj, out_proj])
                    if rnn is not None:
                        all_modules.extend([rnn])
                    for module in all_modules:
                        for n, p in module.named_parameters():
                            if p.grad is not None:
                                if p.grad.isinf().any():
                                    print ('WARNING: inf grad found in rnn!')
                                    print (n)
                                    print ('WARNING: filling with 0! inf grad found!')
                                    p.grad.data[p.grad.isinf()] = 0
                                if p.grad.isnan().any():
                                    print ('WARNING: nan grad found in rnn!')
                                    print (n)
                                    print ('WARNING: filling with 0! inf grad found!')
                                    p.grad.data[p.grad.isnan()] = 0
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_([sigmas], args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                #optimizer_sigmas.step()
                #optimizer_sigmas.zero_grad(set_to_none=True)
            loss = loss.item()
            try:
                ppl = math.exp(loss)
            except Exception as e:
                ppl = float('inf')
            ppl0 = math.exp(nll.item())
            if step % 100 == 0:
                print (f"Step: {step}. PPL: {ppl}. Loss: {loss}. PPL0: {ppl0}. NLL: {nll}. KL: {kl}. Accuracy: {accuracy}")
                print (sigmas)
                sys.stdout.flush()
            step += 1
            #if step >= 2000:
            #    break
    #    accuracy, word_accuracy, ppl = evaluate(model, model_q, train_dataloader, tokenizer, ctx, sigmas)
    #    accuracy, word_accuracy, ppl = evaluate(model, model_q, val_dataloader, tokenizer, ctx, sigmas)
        #ppl, loss, ppl_nll, loss_nll, loss_kl, accuracy = evaluate(model, model_q, val_dataloader, tokenizer, ctx, sigmas, mlps, args.mode)
        #print (f"Val. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Accuracy: {accuracy}")
        ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(args.no_mixture, key_proj, query_proj, out_proj, args.softmax_p, args.softmax_p_temp, args.mult_p, model, model_q, val_dataloader, tokenizer, ctx, sigmas, mlps, mixture_components, rnn, args.mode, mlps_patch)
        print (f"Val. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")
        ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(args.no_mixture, key_proj, query_proj, out_proj, args.softmax_p, args.softmax_p_temp, args.mult_p, model, model_q, test_dataloader, tokenizer, ctx, sigmas, mlps, mixture_components, rnn, args.mode, mlps_patch)
        print (f"Test. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Test Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")

        if eval_dataloader is not None:
            ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(args.no_mixture, key_proj, query_proj, out_proj, args.softmax_p, args.softmax_p_temp, args.mult_p, model, model_q, eval_dataloader, tokenizer, ctx, sigmas, mlps, mixture_components, rnn, args.mode, mlps_patch)
            print (f"Eval PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Eval Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")
        #print (f'Epoch {epoch}. Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
        #print ('sigmas', sigmas)
        sys.stdout.flush()
    #    model.train()
    #    model_q.train()
        model.eval()
        model_q.eval()

if __name__ == "__main__":
    main()
