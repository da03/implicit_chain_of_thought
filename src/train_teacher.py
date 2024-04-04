import math
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse
import os
import sys
import tqdm
import inspect
import logging

from models.teacher import Teacher
from models.configuration_teacher import TeacherConfig
from data import CoTDataset, CoTDataCollator, extract_answer

from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, max_new_tokens):
    teacher.eval()
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    for batch in tqdm.tqdm(dataloader):
        input_ids_all = batch['input_ids_all'].to(device)
        input_ids = input_ids_all
        sep_position_1 = get_sep_position(input_ids, tokenizer.eos_token_id)[0]
        sep_position_2 = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)[0]
        sep_position_3 = get_sep_position(input_ids, tokenizer.eos_token_id, skip=2)[0]
        sep_position_4 = get_sep_position(input_ids, tokenizer.eos_token_id, skip=3)[0]
        sep_position_5 = get_sep_position(input_ids, tokenizer.eos_token_id, skip=4)[0]
        ans_len_diff = sep_position_5 - sep_position_4 - (input_ids.shape[-1] - 1 - sep_position_5)
        input_ids_problem_1 = torch.cat((input_ids[:, :sep_position_1+1], input_ids[:, sep_position_2+1:sep_position_3+1], input_ids[:, sep_position_4+1:sep_position_5+1]), dim=-1)
        input_ids_problem_2 = torch.cat((input_ids[:, sep_position_1+1:sep_position_2+1], input_ids[:, sep_position_3+1:sep_position_4+1], input_ids[:, sep_position_4+1:sep_position_4+1+ans_len_diff],  input_ids[:, sep_position_5+1:]), dim=-1)
        input_ids = torch.cat((input_ids_problem_1, input_ids_problem_2), dim=0)
        input_ids_all = input_ids
        labels = input_ids.data.clone()
        sep_position =  get_sep_position(input_ids, tokenizer.eos_token_id)[0]
        labels[:, :sep_position+1] = -100
        #labels = batch['labels_all'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        batch_size = input_ids.shape[0]
        with ctx:
            outputs = teacher.compute_loss(input_ids=input_ids_all, labels=labels)
        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        beam_output = teacher.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        # Evaluate
        #import pdb; pdb.set_trace()
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
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    args = parser.parse_args()

    print (args)

    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create Teacher 
    config = TeacherConfig(base_model=args.base_model)
    teacher = Teacher(config).to(device).to(ptdtype)

    # Load data
    tokenizer = teacher.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    trainable_params = list(teacher.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    teacher.train()

    # Train
    step = 0
    prev_token_accuracy = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        teacher.train()
        for batch in tqdm.tqdm(train_dataloader):
            #import pdb; pdb.set_trace()
             #0 7 5 8 * 8 7 4 7<|endoftext|>6 3 4 4 * 7 2 0 4 <|endoftext|> 0 6 5 8 6 + 0 0 9 9 9 5 ( 0 6 4 8 6 6 ) + 0 0 0 8 2 4 3 ( 0 6 4 6 9 0 4 ) + 0 0 0 0 9 9 9 5<|endoftext|>2 5 0 1 3 + 0 2 7 8 8 0 ( 2 7 7 9 1 1 ) + 0 0 0 0 0 0 0 ( 2 7 7 9 1 1 0 ) + 0 0 0 4 4 7 7 1 <|endoftext|> #### 0 6 4 6 8 0 4 6<|endoftext|>2 7 7 3 6 8 7 1 <|endoftext|>
            input_ids = batch['input_ids_all'].to(device)
            sep_position_1 = get_sep_position(input_ids, tokenizer.eos_token_id)[0]
            sep_position_2 = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)[0]
            sep_position_3 = get_sep_position(input_ids, tokenizer.eos_token_id, skip=2)[0]
            sep_position_4 = get_sep_position(input_ids, tokenizer.eos_token_id, skip=3)[0]
            sep_position_5 = get_sep_position(input_ids, tokenizer.eos_token_id, skip=4)[0]
            ans_len_diff = sep_position_5 - sep_position_4 - (input_ids.shape[-1] - 1 - sep_position_5)
            input_ids_problem_1 = torch.cat((input_ids[:, :sep_position_1+1], input_ids[:, sep_position_2+1:sep_position_3+1], input_ids[:, sep_position_4+1:sep_position_5+1]), dim=-1)
            input_ids_problem_2 = torch.cat((input_ids[:, sep_position_1+1:sep_position_2+1], input_ids[:, sep_position_3+1:sep_position_4+1], input_ids[:, sep_position_4+1:sep_position_4+1+ans_len_diff],  input_ids[:, sep_position_5+1:]), dim=-1)
            input_ids = torch.cat((input_ids_problem_1, input_ids_problem_2), dim=0)
            labels = input_ids.data.clone()
            sep_position =  get_sep_position(input_ids, tokenizer.eos_token_id)[0]
            labels[:, :sep_position+1] = -100
            #labels = batch['labels_all'].to(device)
            with ctx:
                outputs = teacher.compute_loss(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()
            if token_accuracy == 1 and prev_token_accuracy == 1:
                print (f"Break Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
                break
            prev_token_accuracy = token_accuracy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            ppl = loss.exp().item()
            if step % 100 == 0:
                print (f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
                sys.stdout.flush()
            step += 1
        accuracy, token_accuracy, ppl = evaluate(val_dataloader, tokenizer, ctx, teacher, args.max_new_tokens)
        print (f'Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        teacher.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))

if __name__ == "__main__":
    main()
