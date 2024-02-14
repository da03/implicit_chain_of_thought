import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse
import os
import sys
import inspect
import tqdm
import logging
import random
import torch.nn as nn

from data import CoTDataset, CoTDataCollator
from models.teacher import Teacher
from models.emulator import Emulator
from models.configuration_emulator import EmulatorConfig


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
random.seed(1234)
torch.manual_seed(1234)
logging.disable(logging.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, emulator, delta, subset, mixture_component_supervision_weight, mixture_supervision_offset):
    total_instances = 0
    total_loss = 0
    total_mixture_component_loss = 0
    for batch in tqdm.tqdm(dataloader):
        #import pdb; pdb.set_trace()
        input_ids_cot = batch['input_ids_cot'].to(device)
        batch_size = input_ids_cot.shape[0]
        with ctx:
            teacher_states, positions_to_extract_per_layer = teacher.extract_states(input_ids=input_ids_cot, delta=delta, subset=subset, return_positions=True)
            if mixture_component_supervision_weight == 0:
                outputs = emulator.compute_loss(input_ids=input_ids_nocot, teacher_states=teacher_states)
            else:
                #import pdb; pdb.set_trace()
                positions_to_extract_per_layer = positions_to_extract_per_layer + mixture_supervision_offset
                supervised_mixture_components = input_ids_cot.gather(1, positions_to_extract_per_layer)
                outputs = emulator.compute_loss(input_ids=input_ids_cot, teacher_states=teacher_states, \
                        supervised_mixture_components=supervised_mixture_components, mixture_component_supervision_weight=mixture_component_supervision_weight)
            #outputs = emulator.compute_loss(input_ids=input_ids_cot, teacher_states=teacher_states)
            loss = outputs.loss
        total_loss += outputs.total_loss.item()
        total_mixture_component_loss += outputs.total_mixture_component_loss
        total_instances += batch_size

    loss = total_loss / total_instances
    mixture_component_loss = total_mixture_component_loss / total_instances / teacher.num_layers
    return loss, mixture_component_loss.exp()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, required=True)
    parser.add_argument('--delta', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--subset', type=str, choices=['diagonal', 'last_column', 'top_row', 'bottom_row', 'first_column'], default='diagonal')
    parser.add_argument('--mixture_size', type=str, default='1')
    parser.add_argument('--mixture_component_supervision_weight', type=float, default=0)
    parser.add_argument('--nopretrain', dest='nopretrain', action='store_true')
    parser.add_argument('--mixture_supervision_offset', type=int, default=0)
    args = parser.parse_args()

    print (args)
    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create Emulator
    config = EmulatorConfig(base_model=args.base_model, mixture_size=args.mixture_size)
    emulator = Emulator(config, nopretrain=args.nopretrain).to(device).to(ptdtype)

    # Load Teacher
    teacher = Teacher.from_pretrained(args.teacher).to(device).to(ptdtype)

    # Load data
    tokenizer = teacher.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    trainable_params = list(emulator.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    teacher.eval()
    emulator.eval() # to turn off dropout

    for p in teacher.parameters():
        p.requires_grad = False

    # Train
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        for batch in tqdm.tqdm(train_dataloader):
            #import pdb; pdb.set_trace()
            input_ids_cot = batch['input_ids_cot'].to(device)
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            with ctx:
                with torch.no_grad():
                    teacher_states, positions_to_extract_per_layer = teacher.extract_states(input_ids=input_ids_cot, delta=args.delta, subset=args.subset, return_positions=True)
                if args.mixture_component_supervision_weight == 0:
                    outputs = emulator.compute_loss(input_ids=input_ids_nocot, teacher_states=teacher_states)
                else:
                    #import pdb; pdb.set_trace()
                    positions_to_extract_per_layer = positions_to_extract_per_layer + args.mixture_supervision_offset
                    supervised_mixture_components = input_ids_cot.gather(1, positions_to_extract_per_layer)
                    outputs = emulator.compute_loss(input_ids=input_ids_nocot, teacher_states=teacher_states, \
                            supervised_mixture_components=supervised_mixture_components, mixture_component_supervision_weight=args.mixture_component_supervision_weight)
            loss = outputs.loss

            loss.backward()
            if step < 100:
                for n, p in emulator.named_parameters():
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
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                if args.mixture_component_supervision_weight == 0:
                    print (f"Step: {step}. Loss: {loss}.")
                else:
                    mixture_ppl = (outputs.total_mixture_component_loss / input_ids_cot.shape[0] / teacher.num_layers).exp().item()
                    print (f"Step: {step}. Loss: {loss}. Mixture Component PPL: {mixture_ppl}")
                sys.stdout.flush()
            step += 1
        loss, mixture_ppl = evaluate(val_dataloader, tokenizer, ctx, teacher, emulator, args.delta, args.subset, args.mixture_component_supervision_weight, args.mixture_supervision_offset)
        if args.mixture_component_supervision_weight == 0:
            print (f'Val. Loss: {loss}.')
        else:
            print (f'Val. Loss: {loss}. Mixture Component PPL: {mixture_ppl}')
        emulator.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))
    

if __name__ == "__main__":
    main()
