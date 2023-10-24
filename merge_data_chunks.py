import sys, os, glob

input_dir = 'data/distilled_nobrackets_300kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1'

num_partitions = 4

merged_dir = input_dir + '_merged'
os.makedirs(merged_dir, exist_ok=True)
d = {}
for i in range(num_partitions):
    output_dir = input_dir + f'_{i}'
    filenames = glob.glob(os.path.join(output_dir, 'src1_*'))
    for filename in filenames:
        filename_base = os.path.basename(filename)
        with open(filename) as fin:
            lines = fin.readlines()
            if filename_base not in d:
                assert i == 0
                d[filename_base] = []
            d[filename_base].extend(lines)

for filename_base in d:
    filename_out = os.path.join(merged_dir, filename_base)
    with open(filename_out, 'w') as fout:
        lines = d[filename_base]
        for line in lines:
            fout.write(line)
