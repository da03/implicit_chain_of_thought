import sys, os
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
a = []
b = []
c = []
d = []
with open(sys.argv[1]) as fin:
    for line in fin:
        items = line.strip().split('||')
        input, cot_and_output = items
        items = cot_and_output.split(' #### ')
        cot, output = items
        input = input.strip()
        cot = cot.strip()
        output = output.strip()
        #import pdb; pdb.set_trace()
        tokens = tokenizer(input)['input_ids']
        a.append(len(tokens))
        tokens = tokenizer(cot)['input_ids']
        b.append(len(tokens))
        tokens = tokenizer(output)['input_ids']
        c.append(len(tokens))
        tokens = tokenizer(line)['input_ids']
        d.append(len(tokens))
a = np.array(a)
b = np.array(b)
c = np.array(c)
d = np.array(d)

print (np.percentile(a, 50))
print (np.percentile(b, 50))
print (np.percentile(c, 50))
print ('===')
print (np.percentile(d, 10))
print (np.percentile(d, 20))
print (np.percentile(d, 30))
print (np.percentile(d, 40))
print (np.percentile(d, 50))
print (np.percentile(d, 60))
print (np.percentile(d, 70))
print (np.percentile(d, 80))
print (np.percentile(d, 90))
print (np.percentile(d, 95))
print (np.percentile(d, 98))
print (np.percentile(d, 99))
