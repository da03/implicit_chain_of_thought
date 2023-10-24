import re
import shutil
import hashlib
import numpy as np
import time
import sys, os, json, random, argparse
import tiktoken
import tqdm
import glob

def main(input_formula_dir, output_formula_dir):
    os.makedirs(output_formula_dir, exist_ok=True)
    for fname in glob.glob(os.path.join(input_formula_dir, 'src1_*')):
        basename = os.path.basename(fname)
    #for split in ['train', 'valid', 'test']:
        #with open(os.path.join(input_formula_dir, f'src1_{split}.txt')) as fin:
        #    with open(os.path.join(output_formula_dir, f'src1_{split}.txt'), 'w') as fout:
        with open(os.path.join(input_formula_dir, basename)) as fin:
            with open(os.path.join(output_formula_dir, basename), 'w') as fout:
                for line in fin:
                    if ' #### #### ' not in line:
                        assert False
                        print ('warning')
                        continue
                    items = line.strip().split(' #### #### ')
                    first, second = items
                    line = first.strip() + ' = ' + second.strip() + ' #### #### ' + second.strip() + '\n'
                    fout.write(line)

def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Augment")
    #parser.add_argument("--input_formula_dir", type=str, default="data/200kaugmented_math_scaffolding_formula", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_200kaugmented_math_scaffolding_formula", help="Output folder")
    #parser.add_argument("--input_formula_dir", type=str, default="data/300kaugmented_math_scaffolding_formula", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_300kaugmented_math_scaffolding_formula", help="Output folder")
    #parser.add_argument("--input_formula_dir", type=str, default="data/400kaugmented_math_scaffolding_formula", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_400kaugmented_math_scaffolding_formula", help="Output folder")
    #parser.add_argument("--input_formula_dir", type=str, default="data/long_mult_4_inter", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_long_mult_4_inter", help="Output folder")
    #parser.add_argument("--input_formula_dir", type=str, default="data/sharps_long_mult_4_inter", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_ans_long_mult_4_inter", help="Output folder")
    parser.add_argument("--input_formula_dir", type=str, default="data/sharps_long_mult_5_inter", help="Output folder")
    parser.add_argument("--output_formula_dir", type=str, default="data/sharps_ans_long_mult_5_inter", help="Output folder")
    #parser.add_argument("--input_formula_dir", type=str, default="data/distilled_200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1", help="Output folder")
    #parser.add_argument("--input_formula_dir", type=str, default="data/distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1", help="Output folder")
    #parser.add_argument("--input_formula_dir", type=str, default="data/distilled_augmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_distilled_augmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1", help="Output folder")
    #parser.add_argument("--input_formula_dir", type=str, default="data/distilled_nobrackets_300kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1_merged", help="Output folder")
    #parser.add_argument("--output_formula_dir", type=str, default="data/sharps_distilled_nobrackets_300kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1_merged", help="Output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    main(input_formula_dir=args.input_formula_dir, \
         output_formula_dir=args.output_formula_dir)
