import sys
import os

input_folder = sys.argv[1]
output_folder = sys.argv[2]

os.makedirs(output_folder, exist_ok=True)

def process(line):
    items = line.strip().split('||')
    assert len(items) == 2
    input, cot_ans = items
    input = input.strip()
    cot, ans = cot_ans.strip().split('####')
    cot = cot.strip()
    ans = ans.strip()
    return input, cot, ans

for split in ['train', 'valid', 'test_bigbench']:
    filename = os.path.join(input_folder, f'{split}.txt')
    lines = open(filename).readlines()
    with open(os.path.join(output_folder, f'{split}.txt'), 'w') as fout:
        for i in range(len(lines) // 2):
            line1 = lines[i*2]
            line2 = lines[i*2+1]
            input1, cot1, ans1 = process(line1)
            input2, cot2, ans2 = process(line2)
            fout.write(f'{input1} ; {input2}||{cot1} ; {cot2} #### {ans1} ; {ans2}\n')
