import math
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse
import os
import inspect
import tqdm
import logging
import random
from itertools import chain

from data import CoTDataset, CoTDataCollator, extract_answer
from models.student import Student
from models.emulator import Emulator
from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
random.seed(1234)
torch.manual_seed(1234)
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, emulator, student, max_new_tokens):
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    for batch in tqdm.tqdm(dataloader):
        #import pdb; pdb.set_trace()
        input_ids_nocot = batch['input_ids_nocot'].to(device)
        labels_nocot = batch['labels_nocot'].to(device)
        batch_size = input_ids_nocot.shape[0]
        with ctx:
            emulated_teacher_states = emulator(input_ids=input_ids_nocot)
            outputs = student.compute_loss(input_ids=input_ids_nocot, labels=labels_nocot, teacher_states=emulated_teacher_states)
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()
        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        with ctx:
            beam_output = student.generate(
                input_ids=input_ids_nocot,
                teacher_states=emulated_teacher_states,
                max_new_tokens=max_new_tokens,
            )

        # Evaluate
        sep_positions = get_sep_position(input_ids_nocot, tokenizer.eos_token_id)
        for i, (input_ids_i, beam_output_i) in enumerate(zip(input_ids_nocot, beam_output)):
            sep_position = sep_positions[i].item()
            tgt = input_ids_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1
            if i == 0:
                print (f'Input: {tokenizer.decode(input_ids_i[:sep_position], skip_special_tokens=True)}')
                print (f'Target: {tgt_text}')
                print (f'Predicted: {pred_text}')
                print ('')
    accuracy = total_correct / total_instances
    token_accuracy = total_correct_tokens / total_tokens
    loss = total_loss / total_tokens
    ppl = math.exp(loss)
    return accuracy, token_accuracy, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emulator', type=str, required=True)
    parser.add_argument('--student', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--softmax_temperature', type=float, default=0.05)
    parser.add_argument('--fix_emulator', dest='fix_emulator', action='store_true')
    parser.set_defaults(fix_emulator=False)
    args = parser.parse_args()

    print (args)
    
    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Load Student
    student = Student.from_pretrained(args.student).to(device).to(ptdtype)

    # Load Emulator
    emulator = Emulator.from_pretrained(args.emulator).to(device).to(ptdtype)

    # Load data
    tokenizer = emulator.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    if args.fix_emulator:
        trainable_params = list(student.parameters())
        for p in emulator.parameters():
            p.requires_grad = False
    else:
        trainable_params = list(student.parameters()) + list(emulator.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    emulator.eval() # to turn off dropout
    student.eval() # to turn off dropout


    # Train
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        for batch in tqdm.tqdm(train_dataloader):
            #import pdb; pdb.set_trace()
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            labels_nocot = batch['labels_nocot'].to(device)
            with ctx:
                emulated_teacher_states = emulator(input_ids_nocot, requires_backward=not args.fix_emulator)
                outputs = student.compute_loss(input_ids=input_ids_nocot, labels=labels_nocot, teacher_states=emulated_teacher_states)
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            ppl = loss.exp().item()
            if step % 100 == 0:
                print (f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
            step += 1
        accuracy, token_accuracy, ppl = evaluate(val_dataloader, tokenizer, ctx, emulator, student, args.max_new_tokens)
        print (f'Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        student.save_pretrained(os.path.join(args.save_model, 'student', f'checkpoint_{epoch}'))
        emulator.save_pretrained(os.path.join(args.save_model, 'emulator',  f'checkpoint_{epoch}'))


if __name__ == "__main__":
    main()
