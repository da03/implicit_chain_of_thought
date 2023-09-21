import os
import re

def extract_kl_values(filepath):
    """
    Extract KL and KL pred values from the given file.

    Args:
    - filepath: path to the log file.

    Returns:
    - A tuple of (min KL value, min KL pred value)
    """
    with open(filepath, 'r') as f:
        content = f.readlines()

    # Regular expression patterns to match KL and KL pred values
    kl_pattern = r'KL: (\d+\.\d+|nan)'
    kl_pred_pattern = r'KL pred: (\d+\.\d+|nan)'

    kl_values = []
    kl_pred_values = []

    for idx, line in enumerate(content[:-1]):
        if content[idx].strip() == "feed: p, use: argmin":
            kl_match = re.search(kl_pattern, content[idx+1])
            kl_pred_match = re.search(kl_pred_pattern, content[idx+1])
            
            if kl_match and kl_match.group(1) != 'nan':
                kl_values.append(float(kl_match.group(1)))
            
            if kl_pred_match and kl_pred_match.group(1) != 'nan':
                kl_pred_values.append(float(kl_pred_match.group(1)))

    return min(kl_values, default=None), min(kl_pred_values, default=None)


def find_minimums_in_logs(base_directory):
    """
    Walk through the given directory and its subdirectories to find the minimum KL and KL pred values in log files.
    """
    min_kl = float('inf')
    min_kl_pred = float('inf')
    folder_with_min_kl = None
    folder_with_min_kl_pred = None

    for dirpath, dirnames, filenames in os.walk(base_directory):
        if '128' in dirpath:
            continue
        #if '64' in dirpath:
        #    continue
        for filename in filenames:
            if filename.startswith("log"):
                kl, kl_pred = extract_kl_values(os.path.join(dirpath, filename))

                if kl is not None and kl < min_kl:
                    min_kl = kl
                    folder_with_min_kl = dirpath

                if kl_pred is not None and kl_pred < min_kl_pred:
                    min_kl_pred = kl_pred
                    folder_with_min_kl_pred = dirpath

    return folder_with_min_kl, min_kl, folder_with_min_kl_pred, min_kl_pred


if __name__ == "__main__":
    base_directory = 'mixture'  # Change this to your actual path
    folder_with_min_kl, min_kl, folder_with_min_kl_pred, min_kl_pred = find_minimums_in_logs(base_directory)

    print(f"Subfolder with minimum KL value: {folder_with_min_kl}. {min_kl}")
    print(f"Subfolder with minimum KL pred value: {folder_with_min_kl_pred}. {min_kl_pred}")

