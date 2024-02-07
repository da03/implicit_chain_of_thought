from dataclasses import dataclass
import os
import copy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def extract_answer(text):
    split_pattern = '####'
    if split_pattern not in text:
        return text.strip().replace(',', '')
    else:
        _, ans = text.strip().split('####', 1)
        ans = '####' + ans
        ans = ans.strip().replace(',', '')
        return ans

def extract_cot(text):
    split_pattern = '####'
    if split_pattern not in text:
        return None
    else:
        cot, _ = text.strip().split('####', 1)
        cot = cot.strip()
        return cot

# We are adding these parts for 2 multiplications 

def extract_both_answers(text):
    split_pattern = ","
    if split_pattern not in text:
        return None
    _, answers = text.strip().split('####', 1)
    ans1, ans2 = answers.strip().split(' , ', 1)
    ans1 = '#### ' + ans1
    ans2 = '#### ' + ans2
    return ans1, ans2

def extract_first_cot(text):
    cot, _ = text.strip().split('####', 1)
    thought_1, thought_2 = cot.strip().split(' , ',1)   
    return thought_1
    pass

def extract_second_cot(text):
    cot, _ = text.strip().split('####', 1)
    # inputs, thought = cot.strip().split(" || ",1)
    # input_1 , input_2 = inputs.strip().split(' , ',1)
    # thought_1, thought_2 = thought.strip().split(' , ',1)
    
    # cot_1 = input_1 + ' || ' + thought_1
    # cot_2 = input_2 + ' || ' + thought_2
       
    # return cot_2
    cot, _ = text.strip().split('####', 1)
    thought_1, thought_2 = cot.strip().split(' , ',1)   
    return thought_2
    pass

def extract_first_input(text):
    input_1 , input_2 = text.strip().split(' , ',1)
    return input_1

def extract_second_input(text):
    input_1 , input_2 = text.strip().split(' , ',1)
    return input_2
# END OF MY PART 

class CoTDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        print (f'Creating features from dataset file at {file_path}')
        bos_tok = tokenizer.bos_token
        eos_tok = tokenizer.eos_token

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        edited_sents_cot_1 = []
        edited_sents_cot_2= []
        
        edited_sents_only_1 = []
        edited_sents_only_2 = []
        
        edited_sents_all_1 = []
        edited_sents_all_2 = []
        
        edited_sents_nocot_1 = []
        edited_sents_nocot_2 = []
        
        for src, tgt in zip(src_lines, tgt_lines):
            #import pdb; pdb.set_trace()
            ans1, _ = extract_both_answers(tgt)
            _ , ans2 = extract_both_answers(tgt)
            cot1 = extract_first_cot(tgt)
            cot2 = extract_second_cot(tgt)
            src1 = extract_first_input(src)
            src2 = extract_second_input(src)
            sent1 = ' {} {} '.format(src1, bos_tok) + cot1 + ' {}'.format(eos_tok)
            sent2 = ' {} {} '.format(src2, bos_tok) + cot2 + ' {}'.format(eos_tok)
            
            edited_sents_cot_1.append(sent1)
            edited_sents_cot_2.append(sent2)
            
            sent1= ' {} {} '.format(src1, bos_tok)
            sent2= ' {} {} '.format(src2, bos_tok)
            
            edited_sents_only_1.append(sent1)
            edited_sents_only_2.append(sent2)
            
            sent1 = ' {} {} '.format(src1, bos_tok) + cot1 + ' {} '.format(eos_tok) + ans1 + ' {}'.format(eos_tok)
            sent2 = ' {} {} '.format(src2, bos_tok) + cot2 + ' {} '.format(eos_tok) + ans2 + ' {}'.format(eos_tok)
            
            edited_sents_all_1.append(sent1)
            edited_sents_all_2.append(sent2)
            
            sent1 = ' {} {} '.format(src1, bos_tok) + ans1 + ' {}'.format(eos_tok)
            sent2 = ' {} {} '.format(src2, bos_tok) + ans2 + ' {}'.format(eos_tok)
            
            edited_sents_nocot_1.append(sent1)
            edited_sents_nocot_2.append(sent2)

        batch_encoding_cot_1 = tokenizer(edited_sents_cot_1, add_special_tokens=True, truncation=True, max_length=max_length)
        batch_encoding_cot_2 = tokenizer(edited_sents_cot_2, add_special_tokens=True, truncation=True, max_length=max_length)       
        batch_encoding_only_1 = tokenizer(edited_sents_only_1, add_special_tokens=True, truncation=True, max_length=max_length)
        batch_encoding_only_2 = tokenizer(edited_sents_only_2, add_special_tokens=True, truncation=True, max_length=max_length)      
        batch_encoding_all_1 = tokenizer(edited_sents_all_1, add_special_tokens=True, truncation=True, max_length=max_length)
        batch_encoding_all_2 = tokenizer(edited_sents_all_2, add_special_tokens=True, truncation=True, max_length=max_length)      
        batch_encoding_nocot_1 = tokenizer(edited_sents_nocot_1, add_special_tokens=True, truncation=True, max_length=max_length)
        batch_encoding_nocot_2 = tokenizer(edited_sents_nocot_2, add_special_tokens=True, truncation=True, max_length=max_length)
        
        self.examples_cot_1 = batch_encoding_cot_1["input_ids"]
        self.examples_cot_2 = batch_encoding_cot_2["input_ids"]
        self.examples_only_1 = batch_encoding_only_1["input_ids"]
        self.examples_only_2 = batch_encoding_only_2["input_ids"]
        self.examples_all_1 = batch_encoding_all_1["input_ids"]
        self.examples_all_2 = batch_encoding_all_2["input_ids"]
        self.examples_nocot_1 = batch_encoding_nocot_1["input_ids"]
        self.examples_nocot_2= batch_encoding_nocot_2["input_ids"]

        self.labels_cot_1 = copy.deepcopy(self.examples_cot_1)
        self.labels_cot_2 = copy.deepcopy(self.examples_cot_2)
        self.labels_all_1 = copy.deepcopy(self.examples_all_1)
        self.labels_all_2 = copy.deepcopy(self.examples_all_2)
        self.labels_cot_shift_1= copy.deepcopy(self.examples_cot_1)
        self.labels_cot_shift_2 = copy.deepcopy(self.examples_cot_2)
        self.labels_nocot_1 = copy.deepcopy(self.examples_nocot_1)
        self.labels_nocot_2 = copy.deepcopy(self.examples_nocot_2)

        self.src_sent_cot_1= []
        self.tgt_sent_cot_1 = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = tokenizer.eos_token_id #tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        for i, elem in enumerate(self.labels_cot_1):
            sep_idx = elem.index(separator) + 1
            self.src_sent_cot_1.append(self.examples_cot_1[i][:sep_idx-1])
            self.tgt_sent_cot_1.append(self.examples_cot_1[i][sep_idx-1:])
            self.labels_cot_1[i][:sep_idx] = [-100] * sep_idx
            assert self.labels_all_1[i][sep_idx-1] == separator
            self.labels_all_1[i][:sep_idx] = [-100] * sep_idx
            self.labels_cot_shift_1[i][:sep_idx-1] = [-100] * (sep_idx-1)
            temp_src_len += sep_idx-1
            temp_tgt_len += len(elem) - (sep_idx-1)
            temp_count += 1
            
        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)
        
        self.src_sent_cot_2= []
        self.tgt_sent_cot_2 = []
        
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = tokenizer.eos_token_id #tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        for i, elem in enumerate(self.labels_cot_2):
            sep_idx = elem.index(separator) + 1
            self.src_sent_cot_2.append(self.examples_cot_2[i][:sep_idx-1])
            self.tgt_sent_cot_2.append(self.examples_cot_2[i][sep_idx-1:])
            self.labels_cot_2[i][:sep_idx] = [-100] * sep_idx
            assert self.labels_all_2[i][sep_idx-1] == separator
            self.labels_all_2[i][:sep_idx] = [-100] * sep_idx
            self.labels_cot_shift_2[i][:sep_idx-1] = [-100] * (sep_idx-1)
            temp_src_len += sep_idx-1
            temp_tgt_len += len(elem) - (sep_idx-1)
            temp_count += 1
            
            
        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)

        self.src_sent_nocot_1 = []
        self.tgt_sent_nocot_1 = []
        
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        for i, elem in enumerate(self.labels_nocot_1):
            sep_idx = elem.index(separator) + 1
            self.src_sent_nocot_1.append(self.examples_nocot_1[i][:sep_idx-1])
            self.tgt_sent_nocot_1.append(self.examples_nocot_1[i][sep_idx-1:])
            self.labels_nocot_1[i][:sep_idx] = [-100] * sep_idx
            temp_src_len += sep_idx-1
            temp_tgt_len += len(elem) - (sep_idx-1)
            temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)

        self.src_sent_nocot_2 = []
        self.tgt_sent_nocot_2= []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        
        for i, elem in enumerate(self.labels_nocot_2):
            sep_idx = elem.index(separator) + 1
            self.src_sent_nocot_2.append(self.examples_nocot_2[i][:sep_idx-1])
            self.tgt_sent_nocot_2.append(self.examples_nocot_2[i][sep_idx-1:])
            self.labels_nocot_2[i][:sep_idx] = [-100] * sep_idx
            temp_src_len += sep_idx-1
            temp_tgt_len += len(elem) - (sep_idx-1)
            temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)
        
        
        print(edited_sents_all_1[0])
        print(self.labels_cot_1[0])
        print(self.labels_nocot_1[0])
        print(self.examples_nocot_1[0])
        print(edited_sents_nocot_1[0])
        print(self.src_sent_nocot_1[0])
        print(self.tgt_sent_nocot_1[0])
        
        print(edited_sents_all_2[0])
        print(self.labels_cot_2[0])
        print(self.labels_nocot_2[0])
        print(self.examples_nocot_2[0])
        print(edited_sents_nocot_2[0])
        print(self.src_sent_nocot_2[0])
        print(self.tgt_sent_nocot_2[0])

    def __len__(self):
        return len(self.examples_cot_1)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples_cot_1[i], dtype=torch.long),
                torch.tensor(self.examples_nocot_1[i], dtype=torch.long),
                torch.tensor(self.labels_cot_1[i], dtype=torch.long),
                torch.tensor(self.labels_cot_shift_1[i], dtype=torch.long),
                torch.tensor(self.labels_nocot_1[i], dtype=torch.long),
                torch.tensor(self.src_sent_cot_1[i], dtype=torch.long),
                torch.tensor(self.src_sent_nocot_1[i], dtype=torch.long),
                torch.tensor(self.tgt_sent_cot_1[i], dtype=torch.long),
                torch.tensor(self.tgt_sent_nocot_1[i], dtype=torch.long),
                torch.tensor(self.examples_only_1[i], dtype=torch.long),
                torch.tensor(self.examples_all_1[i], dtype=torch.long),
                torch.tensor(self.labels_all_1[i], dtype=torch.long),
                torch.tensor(self.examples_cot_2[i], dtype=torch.long),
                torch.tensor(self.examples_nocot_2[i], dtype=torch.long),
                torch.tensor(self.labels_cot_2[i], dtype=torch.long),
                torch.tensor(self.labels_cot_shift_2[i], dtype=torch.long),
                torch.tensor(self.labels_nocot_2[i], dtype=torch.long),
                torch.tensor(self.src_sent_cot_2[i], dtype=torch.long),
                torch.tensor(self.src_sent_nocot_2[i], dtype=torch.long),
                torch.tensor(self.tgt_sent_cot_2[i], dtype=torch.long),
                torch.tensor(self.tgt_sent_nocot_2[i], dtype=torch.long),
                torch.tensor(self.examples_only_2[i], dtype=torch.long),
                torch.tensor(self.examples_all_2[i], dtype=torch.long),
                torch.tensor(self.labels_all_2[i], dtype=torch.long),
                )
@dataclass
class CoTDataCollator:
    """
    VAEData collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        #import pdb; pdb.set_trace()
        input_ids_cot_1, input_ids_nocot_1, labels_cot_1, labels_cot_shift_1, labels_nocot_1, src_cot_1, src_nocot_1, tgt_cot_1, tgt_nocot_1, input_ids_only_1, input_ids_all_1, labels_all_1, input_ids_cot_2, input_ids_nocot_2, labels_cot_2, labels_cot_shift_2, labels_nocot_2, src_cot_2, src_nocot_2, tgt_cot_2, tgt_nocot_2, input_ids_only_2, input_ids_all_2, labels_all_2 = zip(*examples)
        input_ids_cot_1 = self._tensorize_batch(input_ids_cot_1)
        input_ids_cot_1[input_ids_cot_1.lt(0)] = self.tokenizer.eos_token_id
        input_ids_only_1 = self._tensorize_batch(input_ids_only_1)
        input_ids_only_1[input_ids_only_1.lt(0)] = self.tokenizer.eos_token_id
        input_ids_all_1 = self._tensorize_batch(input_ids_all_1)
        input_ids_all_1[input_ids_all_1.lt(0)] = self.tokenizer.eos_token_id
        input_ids_nocot_1 = self._tensorize_batch(input_ids_nocot_1)
        input_ids_nocot_1[input_ids_nocot_1.lt(0)] = self.tokenizer.eos_token_id
        labels_cot_1 = self._tensorize_batch(labels_cot_1)
        labels_all_1 = self._tensorize_batch(labels_all_1)
        labels_cot_shift_1 = self._tensorize_batch(labels_cot_shift_1)
        labels_nocot_1 = self._tensorize_batch(labels_nocot_1)
        #----
        input_ids_cot_2 = self._tensorize_batch(input_ids_cot_2)
        input_ids_cot_2[input_ids_cot_2.lt(0)] = self.tokenizer.eos_token_id
        input_ids_only_2 = self._tensorize_batch(input_ids_only_2)
        input_ids_only_2[input_ids_only_2.lt(0)] = self.tokenizer.eos_token_id
        input_ids_all_2 = self._tensorize_batch(input_ids_all_2)
        input_ids_all_2[input_ids_all_2.lt(0)] = self.tokenizer.eos_token_id
        input_ids_nocot_2 = self._tensorize_batch(input_ids_nocot_2)
        input_ids_nocot_2[input_ids_nocot_2.lt(0)] = self.tokenizer.eos_token_id
        labels_cot_2 = self._tensorize_batch(labels_cot_2)
        labels_all_2 = self._tensorize_batch(labels_all_2)
        labels_cot_shift_2 = self._tensorize_batch(labels_cot_shift_2)
        labels_nocot_2 = self._tensorize_batch(labels_nocot_2)
        
        return {"input_ids_cot_1": input_ids_cot_1, "input_ids_nocot_1": input_ids_nocot_1, "labels_cot_1": labels_cot_1, "labels_cot_shift_1": labels_cot_shift_1, "labels_nocot_1": labels_nocot_1, 'input_ids_only_1': input_ids_only_1, 'input_ids_all_1': input_ids_all_1, 'labels_all_1': labels_all_1, "input_ids_cot_2": input_ids_cot_2, "input_ids_nocot_2": input_ids_nocot_2, "labels_cot_2": labels_cot_2, "labels_cot_shift_2": labels_cot_shift_2, "labels_nocot_2": labels_nocot_2, 'input_ids_only_2': input_ids_only_2, 'input_ids_all_2': input_ids_all_2, 'labels_all_2': labels_all_2}

    def _tensorize_batch(self, examples):
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)
