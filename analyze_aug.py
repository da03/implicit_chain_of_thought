import os

def find_highest_accuracy_in_logs(base_dir):
    highest_accuracies = {}

    for root, dirs, files in os.walk(base_dir):
        for name in files:
            print (highest_accuracies)
            if name == 'log.train':
                full_path = os.path.join(root, name)
                try:
                    with open(full_path, 'r') as file:
                        highest_accuracy = 0
                        flag = True
                        for line in file:
                            if 'Deleting: inf' in line:
                                flag = False
                            if flag:
                                continue
                            if "Val. PPL:" in line:
                                parts = line.split(';')
                                accuracy_str = parts[1].strip().split(':')[1].strip()
                                accuracy = float(accuracy_str)
                                if accuracy > highest_accuracy:
                                    highest_accuracy = accuracy
                        if highest_accuracy > 0:
                            subfolder_name = os.path.basename(root)
                            highest_accuracies[subfolder_name] = highest_accuracy
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")

    return highest_accuracies

# Specify the base directory where your 'train_models/gsm8k/gpt2/' folder is located.
base_dir = './train_models/auggsm8k/gpt2/'

# Get the highest accuracies from log files
highest_accuracies = find_highest_accuracy_in_logs(base_dir)

# Print out each subfolder name and the corresponding highest accuracy
items = []
for subfolder, accuracy in highest_accuracies.items():
    items.append((-accuracy, subfolder))
items = sorted(items)
for accuracy, subfolder in items:
    print(f"{subfolder}: {-accuracy}")

