import os

for split in ['train', 'valid', 'test']:
    filename = f'data/math_scaffolding_formula/src1_{split}.txt'
    filename_out = f'data/math_scaffolding_none/src1_{split}.txt'

    with open(filename) as fin:
        with open(filename_out, 'w') as fout:
            for line in fin:
                a, b = line.split('||')
                items = b.split('####')
                fout.write(f'{a}||####{items[1]}')

