import sys, os, json, re

dirname = 'grade-school-math/grade_school_math/data'
out_dirname = 'processed_gsm8k'
dirname_new = 'data/math_scaffolding_formula'


d = {}

for split in ['train', 'test']:
    filename = f'{dirname}/{split}.jsonl'
    with open(filename) as fin:
        for i, line in enumerate(fin):
            a = json.loads(line.strip())
            question = a['question'].strip()
            #assert question in new_dict, question
            #split_new, i, formula = new_dict[question]
            d[question] = (split, i, line, a['answer'])
            #fout, fout_verify = fouts[split_new]


os.makedirs(out_dirname, exist_ok=True)
out_train_filename = os.path.join(out_dirname, 'train.jsonl')
out_train_verify_filename = os.path.join(out_dirname, 'verify_train.jsonl')
out_valid_filename = os.path.join(out_dirname, 'valid.jsonl')
out_valid_verify_filename = os.path.join(out_dirname, 'verify_valid.jsonl')
out_test_filename = os.path.join(out_dirname, 'test.jsonl')
out_test_verify_filename = os.path.join(out_dirname, 'verify_test.jsonl')

with open(out_train_filename, 'w') as fout_train, open(out_train_verify_filename, 'w') as fout_verify_train, open(out_valid_filename, 'w') as fout_valid, open(out_valid_verify_filename, 'w') as fout_verify_valid, open(out_test_filename, 'w') as fout_test, open(out_test_verify_filename, 'w') as fout_verify_test:
    fouts = { 'train': (fout_train, fout_verify_train),
            'valid': (fout_valid, fout_verify_valid),
            'test': (fout_test, fout_verify_test)
    }

    for split in ['train', 'valid', 'test']:
        filename = f'{dirname_new}/src1_{split}.txt'
        with open(filename) as fin:
            for i, line in enumerate(fin):
                #import pdb; pdb.set_trace()
                items = line.strip().split('||')
                assert len(items) == 2, line
                question = items[0]
                formula = items[1]
                split_orig, _, line_out, answer = d[question]
                fout, fout_verify = fouts[split]
                fout.write(line_out)
                formulas = re.findall(r'(<<[^>]+>>)', answer)
                ans = answer.split('####')[-1].strip()
                # Join the formulas with spaces
                formulas_str = ' '.join(formulas)
                fout_verify.write(question.strip() + '||' + formulas_str + f' #### {ans}\n')
