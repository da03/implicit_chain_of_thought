import os
import re

def add_spaces_to_formula(line):
    # This regular expression captures everything inside << >>
    regex = r'<<([^>]*)>>'
    
    # Function to add spaces between characters inside << >>
    def replacer(match):
        formula = match.group(1)
        spaced_formula = ' '.join(formula)
        return f"<< {spaced_formula} >>"

    # Use re.sub to replace each occurrence in the line
    new_line = re.sub(regex, replacer, line)
    return new_line

def add_math_space(input_filename, output_filename):
    # Read from the input file and write to the output file
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            new_line = add_spaces_to_formula(line)
            outfile.write(new_line)


for split in ['train', 'valid', 'test']:
    input_filename = f'data/math_scaffolding_formula/src1_{split}.txt'
    output_filename = f'data/math_scaffolding_formula_with_spaces/src1_{split}.txt'
    add_math_space(input_filename, output_filename)

