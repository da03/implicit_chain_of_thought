import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse
import os
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
def evaluate(dataloader, tokenizer, ctx, teacher, emulator, delta, subset):
    total_instances = 0
    total_loss = 0
    for batch in tqdm.tqdm(dataloader):
        #import pdb; pdb.set_trace()
        input_ids_cot = batch['input_ids_cot'].to(device)
        batch_size = input_ids_cot.shape[0]
        with ctx:
            teacher_states = teacher.extract_states(input_ids=input_ids_cot, delta=delta, subset=subset)
            outputs = emulator.compute_loss(input_ids=input_ids_cot, teacher_states=teacher_states)
            loss = outputs.loss
        total_loss += outputs.total_loss.item()
        total_instances += batch_size

    loss = total_loss / total_instances
    return loss

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
    parser.add_argument('--mixture_size', type=int, default=1)
    args = parser.parse_args()

    print (args)
    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create Emulator
    config = EmulatorConfig(base_model=args.base_model, mixture_size=args.mixture_size)
    emulator = Emulator(config).to(device).to(ptdtype)

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
                    teacher_states = teacher.extract_states(input_ids=input_ids_cot, delta=args.delta, subset=args.subset)
                outputs = emulator.compute_loss(input_ids=input_ids_nocot, teacher_states=teacher_states)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print (f"Step: {step}. Loss: {loss}.")
            step += 1
        loss = evaluate(val_dataloader, tokenizer, ctx, teacher, emulator, args.delta, args.subset)
        print (f'Val. Loss: {loss}.')
        emulator.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))
    

if __name__ == "__main__":
    main()
