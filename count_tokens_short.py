import sys, os

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
#for n in [4, 5, 6, 7, 8, 9]:
#for n in [10, 11, 12, 13, 14, 15]:
for n in range(4, 21):
    folder = f'data/{n}_by_{n}_mult'
    folder = f'/n/rush_lab/Lab/Users/yuntian/implicit/data/long_mult_{n}_inter_mar2124_short'
    train_file = os.path.join(folder, 'train.txt')
    #if not os.path.exists(train_file):
    #    train_file = os.path.join(folder, 'src1_train.txt')
    items = []
    #import pdb; pdb.set_trace()
    with open(train_file) as fin:
        for line in fin:
            a, b = line.strip().split('||')
            b = b.strip().split('####')[0].strip()
            #b = line.strip()
            items.append(len(tokenizer(b)['input_ids']))
            if len(items) > 1000:
                break
    import numpy as np
    a = np.array(items)
    print (f'N: {n}')
    #for percentile in [5, 50, 90, 95]:
    for percentile in [5, 50, 90, 95]:
        print (f'{percentile} percentile: {np.percentile(a, percentile)}')
