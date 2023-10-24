import sys, os, glob

input_dir = 'data/nobrackets_400kaugmented_math_scaffolding_formula'


filenames = glob.glob(os.path.join(input_dir, 'src1_*'))

num_partitions = 8

d = {}
def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    a = []
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        a.append(l[si:si+(d+1 if i < r else d)])
    return a

for filename in filenames:
    filename_base = os.path.basename(filename)
    with open(filename) as fin:
        lines = fin.readlines()
    items = chunks(lines, num_partitions)
    d[filename_base] = items

for i in range(num_partitions):
    output_dir = input_dir + f'_{i}'
    os.makedirs(output_dir, exist_ok=True)
    for filename_base in d:
        with open(os.path.join(output_dir, filename_base), 'w') as fout:
            lines = d[filename_base][i]
            for line in lines:
                fout.write(line)
