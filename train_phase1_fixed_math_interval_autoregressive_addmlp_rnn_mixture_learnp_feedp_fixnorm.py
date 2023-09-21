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


def save_model(model, mlps, rnn, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(mlps.state_dict(), os.path.join(model_dir, 'mlps.pt'))
    torch.save(rnn.state_dict(), os.path.join(model_dir, 'rnn.pt'))

def extract_answer(text):
    ans = text.strip().replace(',', '')
    return ans

def evaluate(model, model_q, dataloader, tokenizer, ctx, sigmas, mlps, rnn, mode, follow=None, interval_arg=-1, mixture_size=-1, feed='p', use='argmin', last_id_minus=0):#, mlps_patch=None):
    assert feed in ['p', 'q']
    if feed == 'p':
        assert use in ['argmin', 'pred']
    else:
        assert use in ['gt']

    with torch.no_grad():
        model.eval()
        model_q.eval()
        total = 0
        word_correct = 0
        total_correct = 0
        total_loss = 0
        total_loss_nll = 0
        total_loss_kl = 0
        total_loss_kl_pred = 0
        num_layers = len(model.transformer.h)
        total_loss_kls = torch.zeros(num_layers)
        total_loss_kls_pred = torch.zeros(num_layers)
        total_loss_kls_mixture = torch.zeros(num_layers, mixture_size)
        total_loss_kls_mixture_pred = torch.zeros(num_layers, mixture_size)
        total_loss_probs_mixture = torch.zeros(num_layers, mixture_size)
        total_loss_ranks_mixture = torch.zeros(num_layers)
        total_instances = 0
        #for batch in tqdm.tqdm(dataloader):
        for batch in dataloader:
            input_ids_cot = batch['input_ids_cot'].to(device)
            input_ids_nocot = batch['input_ids_nocot'].to(device)
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
                    last_id = last_id - last_id_minus

                    ###layers = torch.arange(start=0, end=num_layers+1)
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
                        hidden_state_relevant = (hidden_states.gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif follow == 'top_row':
                        hidden_state_relevant = (hidden_states_cot[-2].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif follow == 'bottom_row':
                        hidden_state_relevant = (hidden_states_cot[0].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif follow == 'bottom_row_above':
                        hidden_state_relevant = (hidden_states_cot[1].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    else:
                        assert False
                    #zs0.append(hidden_state_relevant)
                    #hidden_state_relevant_list.append(hidden_state_relevant + torch.randn_like(hidden_state_relevant) * sigmas[i])
                    hidden_state_relevant_list.append(hidden_state_relevant)
                zs = hidden_state_relevant_list
                hidden_state_relevant_list_rnn = []
                if feed == 'q':
                    rnn_state = None
                    #import pdb; pdb.set_trace()
                    for z in zs: # z: batch, hidden
                        output, rnn_state = rnn(z.unsqueeze(0), rnn_state)
                        hidden_state_relevant_list_rnn.append(output.squeeze(0))
                    #outputs_nocot = model.forward_zs(input_ids=input_ids_nocot, zs=hidden_state_relevant_list_rnn, first_ids=first_ids, clone=True)
                    outputs_nocot = model.forward_zs(input_ids=input_ids_nocot, zs=hidden_state_relevant_list_rnn, first_ids=first_ids)
                elif feed == 'p':
                    if use == 'argmin':
                        outputs_nocot = model.forward_zs_feedp_argmin(input_ids=input_ids_nocot, zs=hidden_state_relevant_list, first_ids=first_ids, rnn=rnn, mlps=mlps)
                    elif use == 'pred':
                        outputs_nocot = model.forward_zs_feedp_pred(input_ids=input_ids_nocot, zs=None, first_ids=first_ids, rnn=rnn, mlps=mlps)
                    else:
                        assert False
                else:
                    assert False
                zs_p = outputs_nocot.zs_p
                zs_q = hidden_state_relevant_list
                #zs_q = zs0 #outputs_nocot.zs_q
            logits = outputs_nocot.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_nocot[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            kl = 0.
            kl_pred = 0.
            zs_p = get_relevant_zs(zs_p, zs_q, mode)
            kl_i = 0
            for z, zp, sigma_i, mlp in zip(zs_q, zs_p, sigmas, mlps):
                zps = mlp(zp) # bsz, hidden_size x  mixture_size+1
                zps = zps.view(batch_size, hidden_size, -1)
                if mixture_size == 1:
                    zps_pred = zps
                else:
                    zps_pred = zps[:, :, :-1] # bsz, hidden_size, mixture_size
                    c = zps[:, :, -1] # bsz, hidden_size
                    log_probs = torch.bmm(c.unsqueeze(1), zps_pred).squeeze(1).log_softmax(-1) # bsz, mixture_size
                    log_probs_sorted, ids_sorted_pred = log_probs.sort(dim=-1, descending=True)
                    probs_sorted = log_probs_sorted.exp() # bsz, mixture_size
                diff = zps_pred - z.unsqueeze(-1)
                kls = (diff*diff).sum(1) / 2 # bsz, mixture_size

                if False:
                    kls = kls + log_probs
                    kl_item = torch.logsumexp(kls, dim=-1)
                else:
                    kl_item, kl_ids = kls.min(-1) # bsz
                    if mixture_size > 1:
                        log_probs_selected = log_probs.gather(1, kl_ids.view(-1, 1))
                    #ids_mixture[kl_i] = kl_ids
                    ids_sorted = kls.argsort(dim=-1, descending=False)
                kls_sorted = kls.gather(1, ids_sorted)
                if mixture_size > 1:
                    kls_sorted_pred = kls.gather(1, ids_sorted_pred) # bsz, mixture_size
                    kl_item_pred = kls_sorted_pred[:, 0].sum() / labels_nocot.shape[0]  #/ labels_nocot[..., 1:].ge(0).sum().item()
                    probs_sorted = log_probs.gather(1, ids_sorted).exp()
                else:
                    kls_sorted_pred = kls
                    kl_item_pred = kls_sorted_pred[:, 0].sum() / labels_nocot.shape[0]  #/ labels_nocot[..., 1:].ge(0).sum().item()
                total_loss_kls_mixture[kl_i] += kls_sorted.sum(0).cpu()
                total_loss_kls_mixture_pred[kl_i] += kls_sorted_pred.sum(0).cpu()
                if mixture_size > 1:
                    total_loss_probs_mixture[kl_i] += probs_sorted.sum(0).cpu()
                else:
                    total_loss_probs_mixture[kl_i] += batch_size
                if mixture_size > 1:
                    total_loss_ranks_mixture[kl_i] += ids_sorted_pred.eq(kl_ids.view(-1, 1)).float().argmax(-1).sum().cpu()
                else:
                    total_loss_ranks_mixture[kl_i] += 0
                #kl_item = ((z-mlp(zp))*(z-mlp(zp))).sum() / sigma_i / sigma_i / 2 / labels_nocot[..., 1:].ge(0).sum().item()
                kl_item = kl_item.sum() / labels_nocot.shape[0] # labels_nocot[..., 1:].ge(0).sum().item()
                kl += kl_item
                kl_pred += kl_item_pred
                total_loss_kls[kl_i] += kl_item.item() * labels_nocot.shape[0]  #* labels_nocot[...,1:].ge(0).sum().item()
                total_loss_kls_pred[kl_i] += kl_item_pred.item() * labels_nocot.shape[0] #  labels_nocot[...,1:].ge(0).sum().item()
                kl_i += 1
            total_loss += (loss.item() + kl.item()) * labels_nocot.shape[0] #labels_nocot[...,1:].ge(0).sum().item()
            total_loss_kl += kl.item() * labels_nocot.shape[0] # labels_nocot[...,1:].ge(0).sum().item()
            total_loss_kl_pred += kl_pred.item() * labels_nocot.shape[0] # labels_nocot[...,1:].ge(0).sum().item()
            total_loss_nll += loss.item() * labels_nocot.shape[0] # labels_nocot[...,1:].ge(0).sum().item()

            labels_pred = logits.argmax(-1)
            correct = ((labels_pred[...,:-1] == labels_nocot[..., 1:]) * labels_nocot[..., 1:].ge(0)).sum().item()
            word_correct += correct
            total += labels_nocot[..., 1:].ge(0).sum().item()


            # TODO: generate and evaluate accuracy
            # activate beam search and early_stopping
            #import pdb; pdb.set_trace()

            for i, input_ids_single in enumerate(input_ids_nocot):
                total_instances += 1
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
        loss_kl = total_loss_kl / total_instances
        loss_kls = total_loss_kls / total_instances
        loss_kl_pred = total_loss_kl_pred / total_instances
        loss_kls_pred = total_loss_kls_pred / total_instances
        loss_kls_mixture = total_loss_kls_mixture / total_instances
        loss_kls_mixture_pred = total_loss_kls_mixture_pred / total_instances
        loss_probs_mixture = total_loss_probs_mixture / total_instances
        loss_ranks_mixture = total_loss_ranks_mixture / total_instances
    return ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, loss_kls, loss_kls_mixture, loss_kls_mixture_pred, loss_probs_mixture, loss_ranks_mixture, loss_kl_pred, loss_kls_pred

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
    parser.add_argument('--kl_mean_weight', type=float, default=0.01)
    parser.add_argument('--p_mean_weight', type=float, default=0.01)
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--save_model', type=str, default='model_nocot')
    parser.add_argument('--qmodel', type=str, default='gpt2')
    parser.add_argument('--residual', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['top', 'interleave', 'bottom', 'none'], default='none')
    parser.add_argument('--feed', type=str, choices=['q', 'p'], default='p')
    parser.add_argument('--use', type=str, choices=['argmin', 'pred', 'gt'], default='argmin')
    parser.add_argument('--compile', type=int, default=1)
    parser.add_argument('--no_save', type=int, default=0)
    parser.add_argument('--last_id_minus', type=int, default=0)
    parser.add_argument('--interval', type=int, default=-1)
    parser.add_argument('--mixture_size', type=int, default=32)
    parser.add_argument('--follow', type=str, choices=['diagonal_orig', 'diagonal', 'last_column', 'top_row', 'bottom_row', 'bottom_row_above',  'first_column'], default='diagonal')
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
        model_q = torch.compile(model_q)
    else:
        print ('WARNING: no compile!')
    # TODO: maybe use pretrained model here?
    #model.apply(model._init_weights)
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available
    extra_args = dict(fused=True) if use_fused else dict()
    num_layers = len(model.transformer.h)
    sigmas = torch.ones(num_layers).to(ptdtype).to(device)
    #import pdb; pdb.set_trace()
    #mlp: model -> model_q
    hidden_size_in = model.config.hidden_size
    hidden_size_out = model_q.config.hidden_size
    hidden_size_mid = 4 * max(hidden_size_in, hidden_size_out)
    if args.mixture_size == 1:
        a = 1
    else:
        a = args.mixture_size + 1
    mlps = nn.ModuleList([nn.Sequential(
             nn.Linear(hidden_size_in, hidden_size_mid),
             nn.ReLU(),
             nn.Linear(hidden_size_mid, hidden_size_out * a),
             ) for _ in range(num_layers)]).to(device).to(ptdtype)

    rnn = nn.LSTM(input_size=hidden_size_in, hidden_size=hidden_size_in, num_layers=1, batch_first=False, dropout=0, bidirectional=False).to(device).to(ptdtype)
    #mlps_patch = nn.ModuleList([nn.Sequential(
    #         nn.Linear(hidden_size_in, hidden_size_mid),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size_mid, hidden_size_out),
    #         ) for _ in range(num_layers)]).to(device).to(ptdtype)
    #mlps_patch.load_state_dict(torch.load(os.path.join(args.qmodel, 'mlps.pt')))
    #sigmas = torch.zeros(num_layers).to(ptdtype).to(device)
    sigmas = torch.nn.Parameter(sigmas)
    #import pdb; pdb.set_trace()
    #optimizer = torch.optim.AdamW([sigmas] + list(model.parameters())+list(model_q.parameters()), lr=args.lr)
    for p in model_q.parameters():
        p.requires_grad = False
    #for p in mlps_patch.parameters():
    #    p.requires_grad = False
    sigmas.requires_grad = False
    #optimizer = torch.optim.AdamW([sigmas] + list(model.parameters()) + list(mlp.parameters()), lr=args.lr)
    use_fused = True 
    extra_args = dict(fused=True) if use_fused else dict()
    #optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlps.parameters()), lr=args.lr, **extra_args)
    all_parameters = list(model.parameters()) + list(mlps.parameters()) + list(rnn.parameters())
    optimizer = torch.optim.AdamW(all_parameters, lr=args.lr, **extra_args)
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
    for setting, dataloader in [('val', val_dataloader), ('test', test_dataloader)]:
        for feed, use in [('q', 'gt'), ('p', 'argmin'), ('p', 'pred')]: #, (), (), ()]:
            print ('*'*10)
            print (setting)
            print (f'feed: {feed}, use: {use}')
            ppl, loss, ppl_nll, loss_nll, loss_kl, accuracy, loss_kls, loss_kls_mixture, loss_kls_mixture_pred, loss_probs_mixture, loss_ranks_mixture, loss_kl_pred, loss_kls_pred = evaluate(model, model_q, dataloader, tokenizer, ctx, sigmas, mlps, rnn, args.mode, args.follow, interval_arg=args.interval, mixture_size=args.mixture_size, feed=feed, use=use, last_id_minus=args.last_id_minus)#, mlps_patch)
            #return ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, loss_kls, loss_kls_mixture, loss_kls_mixture_pred, loss_probs_mixture, loss_ranks_mixture
            print (f"Val. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Accuracy: {accuracy}. KL pred: {loss_kl_pred}")
            loss_kls = [ '%.2f' % elem for elem in loss_kls.tolist() ]
            print ('kls real', loss_kls)
            loss_kls = [ '%.2f' % elem for elem in loss_kls_pred.tolist() ]
            print ('kls pred', loss_kls)
            for kl_i, (loss_prob_mixture, loss_rank_mixture, loss_kl_mixture, loss_kl_mixture_pred) in enumerate(zip(loss_probs_mixture, loss_ranks_mixture, loss_kls_mixture, loss_kls_mixture_pred)):
                loss_prob_mixture = [ '%.2f' % elem for elem in loss_prob_mixture.tolist() ]
                print (kl_i, loss_prob_mixture)
                loss_kl_mixture = [ '%.2f' % elem for elem in loss_kl_mixture.tolist() ]
                print (kl_i, 'kl sorted', loss_kl_mixture)
                loss_kl_mixture = [ '%.2f' % elem for elem in loss_kl_mixture_pred.tolist() ]
                print (kl_i, 'kl pred', loss_kl_mixture)
                print (kl_i, 'rank pred', loss_rank_mixture)
            print ('='*10)
    #model.train()
    #model_q.train()
    model.eval()
    model_q.eval()

    #model.eval()
    #model_q.eval()
    step = 0
    #import pdb; pdb.set_trace()
    for epoch in range(args.epochs):
        if args.no_save != 1:
            save_model(model, mlps, rnn, tokenizer, f'{args.save_model}/checkpoint_{epoch}_{args.lr}')
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
                        hidden_state_relevant = (hidden_states.gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif args.follow == 'top_row':
                        hidden_state_relevant = (hidden_states_cot[-2].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif args.follow == 'bottom_row':
                        hidden_state_relevant = (hidden_states_cot[0].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    elif args.follow == 'bottom_row_above':
                        hidden_state_relevant = (hidden_states_cot[1].gather(1, relevant_ids[:,i:(i+1)].unsqueeze(-1).expand(-1, -1, hidden_size)).squeeze(1))
                    else:
                        assert False
                    #zs0.append(hidden_state_relevant)
                    #hidden_state_relevant_list.append(hidden_state_relevant + torch.randn_like(hidden_state_relevant) * sigmas[i])
                    hidden_state_relevant_list.append(hidden_state_relevant)

                ###for hidden_states in hidden_states_cot:
                ###    hidden_states[mask] = 0 # batch_size, seq_len, hidden_size
                zs = hidden_state_relevant_list
                if args.feed == 'q':
                    hidden_state_relevant_list_rnn = []
                    rnn_state = None
                    for z in zs: # z: batch, hidden
                        output, rnn_state = rnn(z.unsqueeze(0), rnn_state)
                        hidden_state_relevant_list_rnn.append(output.squeeze(0))
                    outputs_nocot = model.forward_zs(input_ids=input_ids_nocot, zs=hidden_state_relevant_list_rnn, first_ids=first_ids, clone=True)
                elif feed == 'p':
                    if args.use == 'argmin':
                        outputs_nocot = model.forward_zs_feedp_argmin(input_ids=input_ids_nocot, zs=hidden_state_relevant_list, first_ids=first_ids, clone=True, rnn=rnn, mlps=mlps)
                    elif args.use == 'pred':
                        outputs_nocot = model.forward_zs_feedp_pred(input_ids=input_ids_nocot, zs=None, first_ids=first_ids, clone=True, rnn=rnn, mlps=mlps)
                    else:
                        assert False
                else:
                    assert False
                zs_p = outputs_nocot.zs_p
                zs_q = hidden_state_relevant_list
                #zs_q = zs0 #outputs_nocot.zs_q
            #loss = outputs.loss
            logits = outputs_nocot.logits

            labels_pred = logits.argmax(-1)
            #import pdb; pdb.set_trace()
            correct = ((labels_pred[...,:-1] == labels_nocot[...,1:]) * labels_nocot[...,1:].ge(0)).sum().item()
            total = labels_nocot[...,1:].ge(0).sum().item()
            accuracy = correct / total
            total_instances = labels_nocot.shape[0]

            kl = 0.
            #import pdb; pdb.set_trace()
            zs_p = get_relevant_zs(zs_p, zs_q, args.mode)
            kls = torch.zeros(num_layers)
            mixture_size = args.mixture_size
            kls_mixture = torch.zeros(num_layers, mixture_size)
            kls_mixture_pred = torch.zeros(num_layers, mixture_size)
            probs_mixture = torch.zeros(num_layers, mixture_size)
            ids_mixture = torch.zeros(num_layers, batch_size)
            ids_mixture_pred = torch.zeros(num_layers, batch_size)
            kl_i = 0
            extra = 0
            for z, zp, sigma_i, mlp in zip(zs_q, zs_p, sigmas, mlps):
                zps = mlp(zp) # bsz, hidden_size x  mixture_size+1
                zps = zps.view(batch_size, hidden_size, -1)
                if mixture_size == 1:
                    zps_pred = zps
                else:
                    zps_pred = zps[:, :, :-1] # bsz, hidden_size, mixture_size
                    c = zps[:, :, -1] # bsz, hidden_size
                diff = zps_pred - z.unsqueeze(-1)
                kls_ = (diff*diff).sum(1) / 2 # bsz, mixture_size
                #if torch.isnan(kls_).any():
                #    import pdb; pdb.set_trace()
                #if torch.isinf(kls_).any():
                #    import pdb; pdb.set_trace()
                if mixture_size > 1:
                    log_probs = torch.bmm(c.unsqueeze(1), zps_pred).squeeze(1).log_softmax(-1) # bsz, mixture_size
                    #import pdb; pdb.set_trace()
                    with torch.no_grad():
                        log_probs_sorted, ids_sorted_pred = log_probs.sort(dim=-1, descending=True)
                if False:
                    kls_ = kls_ + log_probs
                    kl_item = torch.logsumexp(kls_, dim=-1)
                    extra +=- 0.05 * (log_probs).sum() / batch_size
                else:
                    kl_item, kl_ids = kls_.min(-1) # bsz
                    ids_mixture[kl_i] = kl_ids
                    if mixture_size > 1:
                        ids_mixture_pred[kl_i] = ids_sorted_pred.eq(kl_ids.view(-1, 1)).float().argmax(-1)
                    else:
                        ids_mixture_pred[kl_i] = 0
                    ids_sorted = kls_.argsort(dim=-1, descending=False)
                    if mixture_size > 1:
                        log_probs_selected = log_probs.gather(1, kl_ids.view(-1, 1))
                    if args.p_mean_weight > 0:
                        extra += -args.p_mean_weight * log_probs_selected.mean()
                    #extra += 0.01 * kls_.mean()
                    if args.kl_mean_weight > 0:
                        extra += args.kl_mean_weight * kls_.mean()
                with torch.no_grad():
                    #probs_sorted = log_probs_sorted.exp() # bsz, mixture_size
                    kls_sorted = kls_.gather(1, ids_sorted)
                    if mixture_size == 1:
                        kls_sorted_pred = kls_
                    else:
                        kls_sorted_pred = kls_.gather(1, ids_sorted_pred)
                        probs_sorted = log_probs.gather(1, ids_sorted).exp()
                    kls_mixture[kl_i] = kls_sorted.sum(0).cpu() / total_instances
                    kls_mixture_pred[kl_i] = kls_sorted_pred.sum(0).cpu() / total_instances
                    if mixture_size > 1:
                        probs_mixture[kl_i] = probs_sorted.sum(0).cpu() / probs_sorted.shape[0]
                    else:
                        probs_mixture[kl_i] = 1

                #kl_item = ((z-mlp(zp))*(z-mlp(zp))).sum() / sigma_i / sigma_i / 2 / labels_nocot[..., 1:].ge(0).sum().item()
                kl_item = kl_item.sum() / total_instances
                kl += kl_item
                #total_loss_kls[kl_i] += kl_item.item() * labels_nocot[...,1:].ge(0).sum().item()
                #kl_i += 1
                #kl += ((z-zp)*(z-zp)).sum() / sigma_i / sigma_i / 2 / total
                #kl_item = ((z-mlp(zp))*(z-mlp(zp))).sum() / sigma_i / sigma_i / 2 / total
                #kl += kl_item
                kls[kl_i] += kl_item.item()
                kl_i += 1


            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_nocot[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) 
            loss = nll + kl
            #nll.div(args.accumulate).backward()
            #loss.div(args.accumulate).backward()
            (kl + extra).div(args.accumulate).backward()
            #import pdb; pdb.set_trace()

            if step % args.accumulate == args.accumulate-1:
                ###torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                ####torch.nn.utils.clip_grad_norm_(model_q.parameters(), args.max_grad_norm)
                ###torch.nn.utils.clip_grad_norm_(mlps.parameters(), args.max_grad_norm)
                ###torch.nn.utils.clip_grad_norm_(rnn.parameters(), args.max_grad_norm)

                torch.nn.utils.clip_grad_norm_(all_parameters, args.max_grad_norm)
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
                kls = [ '%.2f' % elem for elem in kls.tolist() ]
                print (kls)
                for kl_i, (loss_prob_mixture, loss_kl_mixture, loss_kl_mixture_pred, id_mixture, id_mixture_pred) in enumerate(zip(probs_mixture, kls_mixture, kls_mixture_pred, ids_mixture, ids_mixture_pred)):
                    loss_prob_mixture = [ '%.2f' % elem for elem in loss_prob_mixture.tolist() ]
                    print (kl_i, loss_prob_mixture)
                    loss_kl_mixture = [ '%.2f' % elem for elem in loss_kl_mixture.tolist() ]
                    print (kl_i, 'min loss', loss_kl_mixture)
                    loss_kl_mixture_pred = [ '%.2f' % elem for elem in loss_kl_mixture_pred.tolist() ]
                    print (kl_i, 'pred loss', loss_kl_mixture_pred)
                    print (kl_i, 'min component', id_mixture)
                    print (kl_i, 'rank', id_mixture_pred)
                sys.stdout.flush()
            step += 1
    #    accuracy, word_accuracy, ppl = evaluate(model, model_q, train_dataloader, tokenizer, ctx, sigmas)
    #    accuracy, word_accuracy, ppl = evaluate(model, model_q, val_dataloader, tokenizer, ctx, sigmas)
        #ppl, loss, ppl_nll, loss_nll, loss_kl, accuracy, loss_kls, loss_kls_mixture, loss_probs_mixture = evaluate(model, model_q, val_dataloader, tokenizer, ctx, sigmas, mlps, rnn, args.mode, args.follow, interval_arg=args.interval, mixture_size=args.mixture_size)#, mlps_patch)
        #print (f"Val. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Accuracy: {accuracy}")
        #loss_kls = [ '%.2f' % elem for elem in loss_kls.tolist() ]
        #print (loss_kls)
        #for kl_i, (loss_prob_mixture, loss_kl_mixture) in enumerate(zip(loss_probs_mixture, loss_kls_mixture)):
        #    loss_prob_mixture = [ '%.2f' % elem for elem in loss_prob_mixture.tolist() ]
        #    print (kl_i, loss_prob_mixture)
        for setting, dataloader in [('val', val_dataloader), ('test', test_dataloader)]:
            for feed, use in [('q', 'gt'), ('p', 'argmin'), ('p', 'pred')]: #, (), (), ()]:
                print ('*'*10)
                print (setting)
                print (f'feed: {feed}, use: {use}')
                ppl, loss, ppl_nll, loss_nll, loss_kl, accuracy, loss_kls, loss_kls_mixture, loss_kls_mixture_pred, loss_probs_mixture, loss_ranks_mixture, loss_kl_pred, loss_kls_pred = evaluate(model, model_q, dataloader, tokenizer, ctx, sigmas, mlps, rnn, args.mode, args.follow, interval_arg=args.interval, mixture_size=args.mixture_size, feed=feed, use=use, last_id_minus=args.last_id_minus)#, mlps_patch)
                #return ppl, loss, ppl_nll, loss_nll, loss_kl, word_accuracy, loss_kls, loss_kls_mixture, loss_kls_mixture_pred, loss_probs_mixture, loss_ranks_mixture
                print (f"Val. PPL: {ppl}. Loss: {loss}. PPL0: {ppl_nll}. NLL: {loss_nll}. KL: {loss_kl}. Accuracy: {accuracy}. KL pred: {loss_kl_pred}")
                loss_kls = [ '%.2f' % elem for elem in loss_kls.tolist() ]
                print ('kls real', loss_kls)
                loss_kls = [ '%.2f' % elem for elem in loss_kls_pred.tolist() ]
                print ('kls pred', loss_kls)
                for kl_i, (loss_prob_mixture, loss_rank_mixture, loss_kl_mixture, loss_kl_mixture_pred) in enumerate(zip(loss_probs_mixture, loss_ranks_mixture, loss_kls_mixture, loss_kls_mixture_pred)):
                    loss_prob_mixture = [ '%.2f' % elem for elem in loss_prob_mixture.tolist() ]
                    print (kl_i, loss_prob_mixture)
                    loss_kl_mixture = [ '%.2f' % elem for elem in loss_kl_mixture.tolist() ]
                    print (kl_i, 'kl sorted', loss_kl_mixture)
                    loss_kl_mixture = [ '%.2f' % elem for elem in loss_kl_mixture_pred.tolist() ]
                    print (kl_i, 'kl pred', loss_kl_mixture)
                    print (kl_i, 'rank pred', loss_rank_mixture)
                print ('='*10)
        #    loss_kl_mixture = [ '%.2f' % elem for elem in loss_kl_mixture.tolist() ]
        #    print (kl_i, loss_kl_mixture)
        #print (f'Epoch {epoch}. Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
        #print ('sigmas', sigmas)
        sys.stdout.flush()
    #    model.train()
    #    model_q.train()
        model.eval()
        model_q.eval()

if __name__ == "__main__":
    main()
