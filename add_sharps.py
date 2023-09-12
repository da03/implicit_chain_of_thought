import re
import shutil
import hashlib
import numpy as np
import time
import sys, os, json, random, argparse
import tiktoken
import tqdm

def main(input_formula_dir, output_formula_dir):
    os.makedirs(output_formula_dir, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(input_formula_dir, f'src1_{split}.txt')) as fin:
            with open(os.path.join(output_formula_dir, f'src1_{split}.txt'), 'w') as fout:
                for line in fin:
                    line = line.replace(' #### ', ' #### #### ')
                    fout.write(line)

def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Augment")
    parser.add_argument("--input_formula_dir", type=str, default="data/augmented_math_scaffolding_formula", help="Output folder")
    parser.add_argument("--output_formula_dir", type=str, default="data/sharps_augmented_math_scaffolding_formula", help="Output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    main(input_formula_dir=args.input_formula_dir, \
         output_formula_dir=args.output_formula_dir)
