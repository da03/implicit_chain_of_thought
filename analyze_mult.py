import os
import glob

def find_highest_accuracy_in_logs(base_dir):
    highest_accuracies = {}
    for dirname in glob.glob(os.path.join(base_dir, '*')):
        #if '12_by_12' not in dirname:
        #    continue
        #if 'debug' not in dirname:
        #    continue
        #if 'binary' not in dirname:
        #    continue
        #if 'cont' not in dirname:
        #    continue
        #if 'cont_debug' not in dirname:
        #    continue
        #if 'cont_binary' not in dirname:
        #    continue
        #if 'binary' not in dirname:
        #    continue
        #if 'cont' in dirname:
        #    continue
        #if 'shiftall_binarytree_' not in dirname:
        #    continue
        if 'newformat' not in dirname and 'sp' not in dirname:
            continue
        highest_accuracies[dirname] = []
        for subdirname in glob.glob(os.path.join(dirname, 'gpt2', '*')):
            filename = os.path.join(subdirname, 'log.train')
            if not os.path.exists(filename):
                continue
            accuracy = None
            with open(filename) as fin:
                num_deleted = 0
                for line in fin:
                    if 'deleted:' in line:
                        num_deleted = int(line.strip().split()[-1])
                    if "Val. PPL:" in line:
                        parts = line.split(';')
                        accuracy_str = parts[1].strip().split(':')[1].strip()
                        accuracy = float(accuracy_str)
                        accuracy_str = parts[2].strip().split(':')[1].strip().rstrip('.')
                        token_accuracy = float(accuracy_str)
                        if accuracy >= 0:
                            highest_accuracies[dirname].append((-num_deleted, -accuracy, -token_accuracy, subdirname))
            if accuracy is None:
                continue
            #highest_accuracies[dirname].append((-num_deleted, -accuracy, subdirname))

        print (f'='*10)
        print (f'{dirname}')
        items_orig = highest_accuracies[dirname]
        for acc_level in [0.95, 0.9, 0.8, .7, .6, .5, .4, .3, .2, .1, .01, 0]:
            print (f'    ** {acc_level}')
            items = sorted([item for item in items_orig if -item[1] >= acc_level])
            for num_deleted, accuracy, token_accuracy, subfolder in items:
                print(f"    - {-num_deleted}: {-accuracy}, {-token_accuracy} {subfolder}")


    return highest_accuracies

# Specify the base directory where your 'train_models/gsm8k/gpt2/' folder is located.
base_dir = './train_models/'

# Get the highest accuracies from log files
highest_accuracies = find_highest_accuracy_in_logs(base_dir)
