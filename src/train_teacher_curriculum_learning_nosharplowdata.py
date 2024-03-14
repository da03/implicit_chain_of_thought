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
from nosharplowdata import CoTDataset, CoTDataCollator, extract_answer

from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batch_ids(input_ids_list, pad_token_id, device, dtype):
    max_seq_len = max([len(item) for item in input_ids_list])
    batch_size = len(input_ids_list)
    input_ids = torch.Tensor(batch_size, max_seq_len).to(dtype).to(device)
    input_ids.fill_(pad_token_id)
    for batch_id in range(batch_size):
        input_ids[batch_id, :len(input_ids_list[batch_id])] = input_ids_list[batch_id]
    return input_ids


def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, max_new_tokens, to_delete, delete_side, delete_eos):
    teacher.eval()
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    #stop_on_two_eos = True
    for batch in tqdm.tqdm(dataloader):
        input_ids_all = batch['input_ids_all'].to(device)
        labels = batch['labels_all'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        batch_size = input_ids.shape[0]
        first_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        #assert all(first_sep_positions == first_sep_positions[0])
        second_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=1)
        #assert all(second_sep_positions == second_sep_positions[0])
        #import pdb; pdb.set_trace()
        if to_delete > 0:
            input_ids_all_tmp = []
            labels_tmp = []
            if delete_side == 'left':
                start_positions = first_sep_positions + 1 # remove from, including
                end_positions = first_sep_positions + 1 + to_delete # remove to, not including
            else:
                assert delete_side == 'right'
                end_positions = second_sep_positions
                start_positions = second_sep_positions - to_delete
            for batch_id in range(input_ids_all.shape[0]):
                start_position = start_positions[batch_id]
                end_position = end_positions[batch_id]
                start_position = max(start_position, first_sep_positions[batch_id]+1)
                if delete_eos:
                    end_position = min(end_position, second_sep_positions[batch_id] + 1)
                else:
                    end_position = min(end_position, second_sep_positions[batch_id])
                input_ids_all_tmp.append(torch.cat((input_ids_all[batch_id, :start_position], input_ids_all[batch_id, end_position:]), dim=-1))
                labels_tmp.append(torch.cat((labels[batch_id, :start_position], labels[batch_id, end_position:]), dim=-1))
            input_ids_all = batch_ids(input_ids_all_tmp, tokenizer.eos_token_id, input_ids_all.device, input_ids_all.dtype)
            labels = batch_ids(labels_tmp, tokenizer.eos_token_id, input_ids.device, input_ids.dtype)
        with ctx:
            outputs = teacher.compute_loss(input_ids=input_ids_all, labels=labels)
        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        if delete_eos:
            stop_on_two_eos = False
        else:
            stop_on_two_eos = True
        beam_output = teacher.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_on_two_eos=stop_on_two_eos,
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
    parser.add_argument('--max_len_train', type=int, default=-1)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--delete_per_epoch', type=float, default=0)
    parser.add_argument('--delete_beyond', type=int, default=-1)
    parser.add_argument('--delete_side', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--delete_type', type=str, choices=['epoch', 'step'], default='epoch')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--delete_eos', action='store_true')
    parser.set_defaults(delete_eos=False)
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.set_defaults(reset_optimizer=False)
    args = parser.parse_args()

    print (args)

    dtype = 'float32'
    if args.base_model == 'meta-llama/Llama-2-7b-hf':
        dtype = 'bfloat16'
    if args.base_model == 'mistralai/Mistral-7B-v0.1':
        dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create Teacher 
    config = TeacherConfig(base_model=args.base_model)
    teacher = Teacher(config).to(device).to(ptdtype)
    if args.tokenizer is not None:
        #import pdb; pdb.set_trace()
        from transformers import AutoTokenizer
        teacher.config.tokenizer_name = args.tokenizer
        teacher.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load data
    tokenizer = teacher.tokenizer

    #if args.base_model == 'mistralai/Mistral-7B-v0.1':
    #    tokenizer.padding_side  = 'right'
    #    print ('PADDING SIDE CHANGED TO RIGHT')
    #    print (tokenizer)
    #import pdb; pdb.set_trace()
    #if tokenizer.eos_token == '#':
    #    import pdb; pdb.set_trace()
    #    print ('WARNING: changing tokenizer\'s eos token to bos token!')
    #    eos_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    #    del tokenizer.added_tokens_decoder[eos_token_id]
    #    tokenizer.eos_token = tokenizer.bos_token
    #    tokenizer.eos_token_id = tokenizer.bos_token_id
    #    pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    #    del tokenizer.added_tokens_decoder[pad_token_id]
    #    tokenizer.pad_token = tokenizer.bos_token
    #    tokenizer.pad_token_id = tokenizer.bos_token_id
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

    #import numpy as np
    #for dataloader in [train_dataloader, val_dataloader]:
    #    deltas = []
    #    for batch in tqdm.tqdm(dataloader):
    #        input_ids = batch['input_ids_all'].to(device)
    #        labels = batch['labels_all'].to(device)
    #        first_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
    #        #assert all(first_sep_positions == first_sep_positions[0])
    #        second_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)
    #        third_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=2)
    #        #deltas.extend((second_sep_positions-first_sep_positions).cpu().tolist())
    #        deltas.extend(third_sep_positions.cpu().tolist())
    #        #if (second_sep_positions-first_sep_positions).max() > 80:
    #        #    import pdb; pdb.set_trace()
    #    deltas = np.array(deltas)
    #    for percentile in [0, 5, 10, 50, 60, 70, 80, 85, 90, 95, 97, 98, 99 100]:
    #        print (f'percentile: {percentile}, {np.percentile(deltas, percentile)}')
    #        #assert all(second_sep_positions == second_sep_positions[0])
    #        #to_delete = epoch
    #sys.exit(1)
    # Train
    step = 0
    steps_per_epoch = len(train_dataloader)


    for epoch in range(args.epochs):
        if args.delete_type == 'epoch':
            steps_per_epoch = len(train_dataloader)
            to_delete = epoch * args.delete_per_epoch
        else:
            steps_per_epoch = len(train_dataloader)
            to_delete = step / steps_per_epoch * args.delete_per_epoch
        to_delete = int(round(to_delete))
        if args.delete_beyond > 0 and to_delete >= args.delete_beyond:
            to_delete = float('inf') # delete all
        print(f"Epoch {epoch}. Deleting: {to_delete}")
        teacher.train()
        #import pdb; pdb.set_trace()
        for batch in tqdm.tqdm(train_dataloader):
            if args.delete_type == 'step':
                prev_to_delete = to_delete
                to_delete = step / steps_per_epoch * args.delete_per_epoch
                to_delete = int(round(to_delete))
                if to_delete > prev_to_delete:
                    print(f" -epoch {epoch}. step {step}. deleting: {to_delete}")
                    if args.reset_optimizer:
                        print ('RESETTING OPTIMIZER')
                        optimizer.zero_grad(set_to_none=True)
                        del optimizer
                        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
                if args.delete_beyond > 0 and to_delete >= args.delete_beyond:
                    to_delete = float('inf') # delete all
            input_ids = batch['input_ids_all'].to(device)
            #import pdb; pdb.set_trace()
            labels = batch['labels_all'].to(device)
            first_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
            #assert all(first_sep_positions == first_sep_positions[0])
            second_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)
            #assert all(second_sep_positions == second_sep_positions[0])
            #to_delete = epoch
            #to_delete = 1000
            #max_to_delete = second_sep_positions[0] - first_sep_positions[0]
            #to_delete = (epoch / args.epochs) * max_to_delete * 2
            #to_delete = int(round(to_delete.item()))

            #import pdb; pdb.set_trace()
            if to_delete > 0:
                input_ids_tmp = []
                labels_tmp = []
                if args.delete_side == 'left':
                    start_positions = first_sep_positions + 1 # remove from, including
                    end_positions = first_sep_positions + 1 + to_delete # remove to, not including
                else:
                    assert args.delete_side == 'right'
                    end_positions = second_sep_positions
                    start_positions = second_sep_positions - to_delete
                for batch_id in range(input_ids.shape[0]):
                    start_position = start_positions[batch_id]
                    end_position = end_positions[batch_id]
                    start_position = max(start_position, first_sep_positions[batch_id]+1)
                    if args.delete_eos:
                        end_position = min(end_position, second_sep_positions[batch_id] + 1)
                    else:
                        end_position = min(end_position, second_sep_positions[batch_id])
                    input_ids_tmp.append(torch.cat((input_ids[batch_id, :start_position], input_ids[batch_id, end_position:]), dim=-1))
                    labels_tmp.append(torch.cat((labels[batch_id, :start_position], labels[batch_id, end_position:]), dim=-1))
                input_ids = batch_ids(input_ids_tmp, tokenizer.eos_token_id, input_ids.device, input_ids.dtype)
                labels = batch_ids(labels_tmp, tokenizer.eos_token_id, input_ids.device, input_ids.dtype)
            #import pdb; pdb.set_trace()
            print (input_ids.shape)
            if args.max_len_train > 0 and input_ids.shape[-1] > args.max_len_train:
                print ('skipped')
                if args.delete_type == 'step':
                    steps_per_epoch -= 1
                sys.stdout.flush()
                continue
            sys.stdout.flush()
           
            with ctx:
                outputs = teacher.compute_loss(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()

            loss.div(args.accumulate).backward()
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            #torch.cuda.empty_cache() 

            ppl = loss.exp().item()
            if step % 100 == 0:
                print (f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
                sys.stdout.flush()
            step += 1
        print (f'deleted: {to_delete}')
        sys.stdout.flush()
        accuracy, token_accuracy, ppl = evaluate(val_dataloader, tokenizer, ctx, teacher, args.max_new_tokens, to_delete, args.delete_side, args.delete_eos)
        print (f'Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        teacher.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))

if __name__ == "__main__":
    main()
