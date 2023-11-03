import math
import time
import re
import torch
import sys
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import argparse
import os
import inspect
import tqdm
from data import CoTDataset, CoTDataCollator, extract_answer
import logging
import random
import torch.nn as nn

from models.emulator import Emulator
from models.student import Student
from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

random.seed(1234)
torch.manual_seed(1234)
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, emulator, student, max_new_tokens):
    total_time = 0
    total_instances = 0
    total_correct = 0

    for batch in tqdm.tqdm(dataloader):
        input_ids_all = batch['input_ids_nocot'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        start_time = time.time()
        with ctx:
            emulated_teacher_states = emulator(input_ids)

            # Generate from student
            beam_output = student.generate(
                input_ids=input_ids,
                teacher_states=emulated_teacher_states,
                max_new_tokens=max_new_tokens,
            )

        # Evaluate
        #import pdb; pdb.set_trace()
        for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
            #sep_position = input_ids_single.tolist().index(tokenizer.eos_token_id)
            sep_position = sep_positions[i].item()
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            #import pdb; pdb.set_trace()
            total_instances += 1
            if ans == pred_ans:
                total_correct += 1
        end_time = time.time()
        total_time += end_time - start_time

    #print (total_time, total_instances, total_instances / total_time)
    throughput = total_instances / total_time
    accuracy = total_correct / total_instances
    return accuracy, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--student_path', type=str, required=True)
    parser.add_argument('--emulator_path', type=str, required=True)
    args = parser.parse_args()

    print (args)
    
    dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)


    # Load Models
    emulator = Emulator.from_pretrained(args.emulator_path).to(device).to(ptdtype)
    student = Student.from_pretrained(args.student_path).to(device).to(ptdtype)
    emulator.eval()
    student.eval()

    # Load data
    tokenizer = emulator.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    test_dataset = CoTDataset(tokenizer, args.test_path, 1024)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    accuracy, throughput  = evaluate(test_dataloader, tokenizer, ctx, emulator, student, args.max_new_tokens)
    print (f"Test Accuracy: {accuracy}. Throughput: {throughput}")


if __name__ == "__main__":
    main()
