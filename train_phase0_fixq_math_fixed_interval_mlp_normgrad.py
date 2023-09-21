import math
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

def set_relevant_zs(mode, zs_p, zs_q):
    #import pdb; pdb.set_trace()
    if mode == 'none':
        pass
    elif mode == 'top':
        diff = len(zs_p) - len(zs_q)
        zs_p[diff:] = zs_q
    elif mode == 'bottom':
        #if len(zs_q) < len(zs_p):
        #    #diff = len(zs_p) - len(zs_q)
        zs_p[:len(zs_q)] = zs_q
    elif mode == 'interleave':
        assert False
        if len(zs_q) < len(zs_p):
            a = len(zs_p) // len(zs_q)
            zs_p = zs_p[(a-1)::a]
    else:
        assert False
    #assert len(zs_p) == len(zs_q)
    #return zs_p
def get_relevant_zs(zs_p, zs_q, mode):
    #import pdb; pdb.set_trace()
    if mode == 'top':
        if len(zs_q) < len(zs_p):
            diff = len(zs_p) - len(zs_q)
            zs_p = zs_p[diff:]
    elif mode == 'interleave':
        if len(zs_q) < len(zs_p):
            a = len(zs_p) // len(zs_q)
            zs_p = zs_p[(a-1)::a]
    else:
        assert False
    assert len(zs_p) == len(zs_q)
    return zs_p

def save_model(model, model_q, mlps, mlps_merge, tokenizer, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    print ('saving', model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(mlps.state_dict(), os.path.join(model_dir, 'mlps.pt'))
    torch.save(mlps_merge.state_dict(), os.path.join(model_dir, 'mlps_merge.pt'))
    model_q_dir = os.path.join(model_dir, 'q')
    os.makedirs(model_q_dir, exist_ok=True)
    print ('saving', model_q_dir)
    model_q.save_pretrained(model_q_dir)
    tokenizer.save_pretrained(model_q_dir)

def extract_answer(text):
    ans = text.strip().replace(',', '')
    return ans

def evaluate(model, model_q, dataloader, tokenizer, ctx, sigmas, mlps, mlps_merge, mode, residual=False, follow=None, use_max=False, interval_arg=-1, last_id_minus=0, arg_max_new_tokens=300):
    with torch.no_grad():
        model.eval()
        model_q.eval()
        total = 0
        word_correct = 0
        total_correct = 0
        total_loss = 0
        total_loss_nll = 0
        total_loss_kl = 0
        total_instances = 0
        num_layers_p = len(model.transformer.h)
        num_layers_q = len(model_q.transformer.h)
        for batch in tqdm.tqdm(dataloader):
            input_ids_cot = batch['input_ids_cot'].to(device)
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            #import pdb; pdb.set_trace()
            labels_cot = batch['labels_cot'].to(device)
            labels_cot_shift = batch['labels_cot_shift'].to(device)
            mask = labels_cot_shift.lt(0)
            labels_nocot = batch['labels_nocot'].to(device)
            with ctx:
                outputs_cot = model_q(input_ids=input_ids_cot, output_hidden_states=True)
                hidden_states_cot = outputs_cot.hidden_states

                # now, calculate q: batch_size, hidden_size
                batch_size = input_ids_cot.shape[0]
                hidden_size = hidden_states_cot[0].shape[-1]
                num_layers = len(hidden_states_cot) - 1
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
                        if follow == 'diagonal_orig':
                            last_id = len(mask_id_list)
                        elif follow == 'diagonal':
                            last_id = len(mask_id_list)
                        else:
                            last_id = len(mask_id_list) - 1
                            last_id -= 1

                    ###layers = torch.arange(start=0, end=num_layers+1)
                    last_id = last_id - last_id_minus
                    #import pdb; pdb.set_trace()
                    layers = torch.arange(start=0, end=num_layers)
                    a = first_id
                    #   ids = torch.round(first_id + layers * (last_id - 1 - first_id) / (num_layers))
                    if follow == 'diagonal' or follow == 'top_row' or follow == 'bottom_row' or follow == 'bottom_row_above':
                        interval = (last_id - 1 - first_id) / (num_layers-1)
                    elif follow == 'diagonal_orig': # todo: rerun experiments with new setting, but don't think this would change things much
                        interval = (last_id - 1 - first_id) / (num_layers)
                    elif follow == 'first_column':
                        interval = 0
                    elif follow == 'last_column':
                        a = last_id - 1
                        interval = 0
                    else:
                        assert False
                    if interval_arg > 0:
                        interval = interval_arg
                    ids = torch.round(a + layers * interval)
                    if interval_arg > 0:
                        ids = ids.clamp(max=last_id - 1)
                    relevant_ids[batch_id] = ids
                #import pdb; pdb.set_trace()

                # time to compute q
                hidden_state_relevant_list = []
                zs0 = []
                for i, hidden_states in enumerate(hidden_states_cot[:-1]):
                    if follow == 'diagonal' or follow == 'diagonal_orig' or follow == 'first_column' or follow == 'last_column':
                        hidden_state_relevant = mlps[i](hidden_states.gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif follow == 'top_row':
                        hidden_state_relevant = mlps[i](hidden_states_cot[-2].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif follow == 'bottom_row':
                        hidden_state_relevant = mlps[i](hidden_states_cot[0].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif follow == 'bottom_row_above':
                        hidden_state_relevant = mlps[i](hidden_states_cot[1].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    else:
                        assert False
                    zs0.append(hidden_state_relevant)
                    #hidden_state_relevant_list.append(hidden_state_relevant + torch.randn_like(hidden_state_relevant) * sigmas[i])
                    hidden_state_relevant_list.append(hidden_state_relevant)
                zs = hidden_state_relevant_list
                zs_model = [None for _ in range(num_layers_p)]
                #import pdb; pdb.set_trace()
                set_relevant_zs(mode, zs_model, zs)
                #outputs_nocot = model.forward_zs(input_ids=input_ids_nocot, zs=zs_model, first_ids=first_ids, residual=False)
                outputs_nocot = model.forward_zs(input_ids=input_ids_nocot, zs=zs_model, first_ids=first_ids, residual=residual, mlps_merge=mlps_merge)
                #outputs_nocot = model.forward_zs_attn(input_ids=input_ids_nocot, attended_to=hidden_states_cot, attended_to_mask=~mask, first_ids=first_ids, sigmas=sigmas)
                zs_p = outputs_nocot.zs_p
                zs_q = zs0 #outputs_nocot.zs_q
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

            labels_pred = logits.argmax(-1)
            correct = ((labels_pred[...,:-1] == labels_nocot[..., 1:]) * labels_nocot[..., 1:].ge(0)).sum().item()
            word_correct += correct
            total += labels_nocot[..., 1:].ge(0).sum().item()


            # TODO: generate and evaluate accuracy
            # activate beam search and early_stopping
            #import pdb; pdb.set_trace()
            #sep_ids = []
            for i, input_ids_single in enumerate(input_ids_nocot):
                sep_id = input_ids_single.tolist().index(tokenizer.eos_token_id)
                #sep_ids.append(sep_id)
            #assert all([item == sep_id for item in sep_ids])
            #import pdb; pdb.set_trace()
                tgt = input_ids_single[sep_id+1:]
                max_new_tokens = tgt.size(0)+10
                max_new_tokens = tgt[tgt.ne(tokenizer.eos_token_id)].size(0)+10
                beam_size = 1
                if not use_max:
                    max_new_tokens = arg_max_new_tokens
                beam_output = model.generate(
                    input_ids=input_ids_single[:sep_id+1].unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    num_beams=beam_size,
                    early_stopping=True,
                    num_return_sequences=1,
                    first_ids=first_ids[i:i+1].repeat_interleave(beam_size, dim=0),
                    zs=[z[i:i+1].repeat_interleave(beam_size, dim=0) if z is not None else z for z in zs_model],
                    residual=residual,
                    mlps_merge=mlps_merge,
                )
                ##src = input_ids_single[:sep_idx+1]
                ##tgt = input_ids_single[sep_idx+1:]

                ##sep_idx = tgt.tolist().index(tokenizer.eos_token_id)
                ##tgt = tgt[:sep_idx]
                ##tgt_text = tokenizer.decode(tgt)
                ###ans = extract_answer(tgt_text)
                ##ans = tgt_text.strip()


                ##beam_output = model.generate(
                ##    input_ids=src.view(1, -1),
                ##    max_new_tokens=100,
                ##    num_beams=5,
                ##    early_stopping=True,
                ##    num_return_sequences=1,
                ##)
                ##
            #import pdb; pdb.set_trace()
            #i = 0
            #for input_ids_single, beam_output_i in zip(input_ids_nocot, beam_output):
                #tgt = tgt[:sep_id]
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
                i += 1

            #for i, input_ids_single in enumerate(input_ids_nocot):
            #    total_instances += 1
        ##TODO        sep_idx = input_ids_single.tolist().index(tokenizer.eos_token_id)
        ##TODO        src = input_ids_single[:sep_idx+1]
        ##TODO        tgt = input_ids_single[sep_idx+1:]

        ##TODO        sep_idx = tgt.tolist().index(tokenizer.eos_token_id)
        ##TODO        tgt = tgt[:sep_idx]
        ##TODO        tgt_text = tokenizer.decode(tgt)
        ##TODO        ans = extract_answer(tgt_text)

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
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--save_model', type=str, default='model_nocot')
    parser.add_argument('--qmodel', type=str, default='gpt2')
    parser.add_argument('--residual', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['top', 'interleave', 'bottom', 'none'], default='none')
    parser.add_argument('--compile', type=int, default=1)
    parser.add_argument('--interval', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--last_id_minus', type=int, default=0)
    parser.add_argument('--follow', type=str, choices=['diagonal', 'diagonal_orig', 'last_column', 'top_row', 'bottom_row', 'bottom_row_above', 'first_column'], default='diagonal_orig')
    args = parser.parse_args()

    print (args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    #if 'gpt2-xl' not in args.model:
    #    dtype = 'float32'
    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    print (ptdtype, dtype)
    model_q = AutoModelForCausalLM.from_pretrained(args.qmodel).to(device).to(ptdtype)
    print (args.qmodel)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).to(ptdtype)
    if args.compile == 1:
        model = torch.compile(model)
        model_q = torch.compile(model_q)
    else:
        print ('WARNING: no compile!')
    # TODO: maybe use pretrained model here?
    #model.apply(model._init_weights)
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available
    extra_args = dict(fused=True) if use_fused else dict()
    num_layers = len(model.transformer.h)
    num_layers_p = num_layers
    num_layers_q = len(model_q.transformer.h)
    sigmas = torch.ones(num_layers_q).to(ptdtype).to(device)
    hidden_size_in = model_q.config.hidden_size
    hidden_size_out = model.config.hidden_size
    hidden_size_mid = 4 * max(hidden_size_in, hidden_size_out)
    mlps = nn.ModuleList([nn.Sequential(
             nn.Linear(hidden_size_in, hidden_size_mid),
             nn.ReLU(),
             nn.Linear(hidden_size_mid, hidden_size_out),
             ) for _ in range(num_layers_q)]).to(device).to(ptdtype)
    if args.residual == 1:
        a = 1
    else:
        a = 2
    mlps_merge = nn.ModuleList([nn.Sequential(
             nn.Linear(a*hidden_size_in, hidden_size_mid),
             nn.ReLU(),
             nn.Linear(hidden_size_mid, hidden_size_out),
             ) for _ in range(num_layers_q)]).to(device).to(ptdtype)
    #mlps.load_state_dict(torch.load(os.path.join(args.qmodel, 'mlps.pt')))
    #sigmas = torch.zeros(num_layers).to(ptdtype).to(device)
    sigmas = torch.nn.Parameter(sigmas)
    #import pdb; pdb.set_trace()
    #optimizer = torch.optim.AdamW([sigmas] + list(model.parameters())+list(model_q.parameters()), lr=args.lr)
    for p in model_q.parameters():
        p.requires_grad = False
    sigmas.requires_grad = False
    #optimizer = torch.optim.AdamW([sigmas] + list(model.parameters()) + list(mlp.parameters()), lr=args.lr)
    #optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlp.parameters()), lr=args.lr)
    #optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlps.parameters()) + list(model_q.parameters()), lr=args.lr)
    use_fused = True 
    extra_args = dict(fused=True) if use_fused else dict()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, **extra_args)
    all_params = list(model.parameters()) + list(mlps.parameters()) + list(mlps_merge.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, **extra_args)
    #optimizer = torch.optim.AdamW([sigmas] + list(model.parameters())+list(model_q.parameters()), lr=args.lr)
    #optimizer_sigmas = torch.optim.SGD([sigmas], lr=args.lr)
    #optimizer = torch.optim.SGD([sigmas] + list(model.parameters())+list(model_q.parameters()), lr=args.lr)

    collate_fn = VAEDataCollator(tokenizer)
    train_dataset = CoTVAEDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTVAEDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    test_dataset = CoTVAEDataset(tokenizer, args.test_path, 1024)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    #compile = True # use PyTorch 2.0 to compile the model to be faster
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)


    #accuracy, word_accuracy, ppl = evaluate(model, model_q, val_dataloader, tokenizer, ctx, sigmas)
    #print (f'Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
    model.eval()
    model_q.eval()
    ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(model, model_q, val_dataloader, tokenizer, ctx, sigmas, mlps, mlps_merge, args.mode, args.residual==1, args.follow, use_max=True, interval_arg=args.interval, last_id_minus=args.last_id_minus)
    print (f"Val. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")
    ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(model, model_q, test_dataloader, tokenizer, ctx, sigmas, mlps, mlps_merge, args.mode, args.residual==1, args.follow, use_max=True, interval_arg=args.interval, last_id_minus=args.last_id_minus)
    print (f"Test. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Test Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")
    #model.train()
    #model_q.train()

    #model.eval()
    #model_q.eval()
    step = 0
    #import pdb; pdb.set_trace()
    for epoch in range(args.epochs):
        save_model(model, model_q, mlps, mlps_merge, tokenizer, f'{args.save_model}/checkpoint_{epoch}_{args.lr}')
        print(f"Epoch {epoch}") #TODO change epoch

        #model.save_pretrained("finetuned_gpt2")
        for batch in tqdm.tqdm(train_dataloader):
            #import pdb; pdb.set_trace()
            input_ids_cot = batch['input_ids_cot'].to(device)
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            labels_cot = batch['labels_cot'].to(device)
            labels_cot_shift = batch['labels_cot_shift'].to(device)
            mask = labels_cot_shift.lt(0)
            labels_nocot = batch['labels_nocot'].to(device)
            with ctx:
                with torch.no_grad():
                    outputs_cot = model_q(input_ids=input_ids_cot, output_hidden_states=True)
                hidden_states_cot = outputs_cot.hidden_states

                # now, calculate q: batch_size, hidden_size
                batch_size = input_ids_cot.shape[0]
                hidden_size = hidden_states_cot[0].shape[-1]
                num_layers = len(hidden_states_cot) - 1
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
                        if args.follow == 'diagonal_orig':
                            last_id = len(mask_id_list)
                        elif args.follow == 'diagonal':
                            last_id = len(mask_id_list)
                        else:
                            last_id = len(mask_id_list) - 1
                            last_id -= 1
                    last_id = last_id - args.last_id_minus

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
                    ids = torch.round(a + layers * interval)
                    if args.interval > 0:
                        ids = ids.clamp(max=last_id - 1)
                    if step == 0 and batch_id == 0:
                        print ('first id', first_id, 'last_id', last_id)
                        print ('ids', ids)
                        if args.follow != 'diagonal_orig':
                            print ('WARNING: last id is manually subtracting 1 to remove the trailing space!')
                        else:
                            print ('WARNING: future experiments should try to use fixed setting!')
                    relevant_ids[batch_id] = ids
                #import pdb; pdb.set_trace()

                # time to compute q
                hidden_state_relevant_list = []
                zs0 = []
                for i, hidden_states in enumerate(hidden_states_cot[:-1]):
                    if args.follow == 'diagonal' or args.follow == 'diagonal_orig' or args.follow == 'first_column' or args.follow == 'last_column':
                        hidden_state_relevant = mlps[i](hidden_states.gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif args.follow == 'top_row':
                        hidden_state_relevant = mlps[i](hidden_states_cot[-2].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif args.follow == 'bottom_row':
                        hidden_state_relevant = mlps[i](hidden_states_cot[0].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif args.follow == 'bottom_row_above':
                        hidden_state_relevant = mlps[i](hidden_states_cot[1].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    else:
                        assert False
                    zs0.append(hidden_state_relevant)
                    #hidden_state_relevant_list.append(hidden_state_relevant + torch.randn_like(hidden_state_relevant) * sigmas[i])
                    hidden_state_relevant_list.append(hidden_state_relevant)

                ###for hidden_states in hidden_states_cot:
                ###    hidden_states[mask] = 0 # batch_size, seq_len, hidden_size
                zs = hidden_state_relevant_list
                #import pdb; pdb.set_trace()
                zs_model = [None for _ in range(num_layers_p)]
                set_relevant_zs(args.mode, zs_model, zs)
                if args.residual == 1:
                    clone = True
                else:
                    clone = False
                outputs_nocot = model.forward_zs(input_ids=input_ids_nocot, zs=zs_model, first_ids=first_ids, residual=args.residual==1, mlps_merge=mlps_merge, clone=clone)
                #outputs_nocot = model.forward_zs_attn(input_ids=input_ids_nocot, attended_to=hidden_states_cot, attended_to_mask=~mask, first_ids=first_ids, sigmas=sigmas)
                zs_p = outputs_nocot.zs_p
                zs_q = zs0 #outputs_nocot.zs_q
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
            nll.div(args.accumulate).backward()
            #loss.div(args.accumulate).backward()
            #kl.div(args.accumulate).backward()
            #import pdb; pdb.set_trace()

            if step % args.accumulate == args.accumulate-1:
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(model_q.parameters(), args.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(mlps.parameters(), args.max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(mlps_merge.parameters(), args.max_grad_norm)
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
        ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(model, model_q, val_dataloader, tokenizer, ctx, sigmas, mlps, mlps_merge, args.mode, args.residual==1, args.follow, interval_arg=args.interval, last_id_minus=args.last_id_minus, arg_max_new_tokens=args.max_new_tokens)
        print (f"Val. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")
        ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, accuracy = evaluate(model, model_q, test_dataloader, tokenizer, ctx, sigmas, mlps, mlps_merge, args.mode, args.residual==1, args.follow, interval_arg=args.interval, last_id_minus=args.last_id_minus)
        print (f"Test. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Test Accuracy: {accuracy}. Word Accuracy: {word_accuracy}")
        #print (f'Epoch {epoch}. Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
        #print ('sigmas', sigmas)
        sys.stdout.flush()
    #    model.train()
    #    model_q.train()
        model.eval()
        model_q.eval()

if __name__ == "__main__":
    main()
