import sys, os


lines1 = open('data/gsm8k_fixed/src1_train.txt').readlines()
lines2 = open('data/gpt3.5_augmented_math_scaffolding_formula/src1_train.txt').readlines()
#import pdb; pdb.set_trace()
lines = lines1 + lines2

import random
random.seed(1234)
random.shuffle(lines)

with open('data/auggsm8k/train.txt', 'w') as fout:
    for line in lines:
        fout.write(line)
