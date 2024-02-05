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
from models.teacher import Teacher
from models.configuration_teacher import TeacherConfig

dtype = 'float32'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = TeacherConfig(base_model='gpt2')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]


teacher = Teacher(config).to(device).to(ptdtype)
tokenizer = teacher.tokenizer
collate_fn = CoTDataCollator(tokenizer)
train_dataset = CoTDataset(tokenizer, "/home/msghol/implicit_chain_of_thought/data/4_by_4_mult/valid 2x2.txt", 1024)
train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

for batch in tqdm.tqdm(train_dataloader):
     #import pdb; pdb.set_trace()
     input_ids_cot_1 = batch['input_ids_cot_1'].to(device)
     input_ids_cot_2 = batch['input_ids_cot_2'].to(device)
     
     print(input_ids_cot_1)
     print(input_ids_cot_2)
