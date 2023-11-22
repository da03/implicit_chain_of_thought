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

from data import CoTDataset, CoTDataCollator, extract_answer
from models.teacher import Teacher
from models.autoencoder import AutoEncoder
from models.student import Student
from models.configuration_student import StudentConfig
from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

random.seed(1234)
torch.manual_seed(1234)
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, autoencoder, student, delta, subset, max_new_tokens):
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    for batch in tqdm.tqdm(dataloader):
        #import pdb; pdb.set_trace()
        input_ids_all = batch['input_ids_all'].to(device)
        input_ids_nocot = batch['input_ids_nocot'].to(device)
        labels_nocot = batch['labels_nocot'].to(device)
        batch_size = input_ids_nocot.shape[0]
        with ctx:
            teacher_states = teacher.extract_states(input_ids=input_ids_all, delta=delta, subset=None)
            teacher_states_cat = torch.stack(teacher_states, dim=-2) # bsz, seq_len, layers, hidden
            encoded_states = autoencoder.encode(teacher_states_cat)
            teacher_states = encoded_states.transpose(0, 1)
            outputs = student.compute_loss(input_ids=input_ids_nocot, labels=labels_nocot, teacher_states=teacher_states)
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
                teacher_states=teacher_states,
                max_new_tokens=max_new_tokens,
            )

        # Evaluate
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
            sep_position = sep_positions[i].item()
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1
            if i == 0:
                print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
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
    parser.add_argument('--teacher', type=str, required=True)
    parser.add_argument('--autoencoder', type=str, required=True)
    parser.add_argument('--delta', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--subset', type=str, choices=['diagonal', 'last_column', 'top_row', 'bottom_row', 'first_column'], default='diagonal')
    args = parser.parse_args()

    print (args)
    
    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create Student
    config = StudentConfig(base_model=args.base_model)
    student = Student(config).to(device).to(ptdtype)

    # Load Teacher
    teacher = Teacher.from_pretrained(args.teacher).to(device).to(ptdtype)
    autoencoder = AutoEncoder.from_pretrained(args.autoencoder).to(device).to(ptdtype)

    # Load data
    tokenizer = teacher.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    trainable_params = list(student.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    teacher.eval()
    student.eval() # to turn off dropout

    for p in teacher.parameters():
        p.requires_grad = False

    # Train
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        for batch in tqdm.tqdm(train_dataloader):
            #import pdb; pdb.set_trace()
            input_ids_all = batch['input_ids_all'].to(device)
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            labels_nocot = batch['labels_nocot'].to(device)
            with ctx:
                with torch.no_grad():
                    teacher_states = teacher.extract_states(input_ids=input_ids_all, delta=args.delta, subset=None)
                    teacher_states_cat = torch.stack(teacher_states, dim=-2) # bsz, seq_len, layers, hidden
                    encoded_states = autoencoder.encode(teacher_states_cat)
                    teacher_states = encoded_states.transpose(0, 1)
                outputs = student.compute_loss(input_ids=input_ids_nocot, labels=labels_nocot, teacher_states=teacher_states)
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
        accuracy, token_accuracy, ppl = evaluate(val_dataloader, tokenizer, ctx, teacher, autoencoder, student, args.delta, args.subset, args.max_new_tokens)
        print (f'Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        student.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))

if __name__ == "__main__":
    main()
