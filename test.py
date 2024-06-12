with open('data/auggsm8k/train.txt', encoding="utf-8") as fin:
    lines = fin.read().splitlines()
#with open('data/gsm8k_fixed/src1_train.txt') as fin:
    for line in lines:
        if '####' not in line:
            import pdb; pdb.set_trace()
