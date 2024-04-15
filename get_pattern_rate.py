# This file aims to get patterns rate from training set (cleaned).

from concat_pattern_password import get_pattern
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="path of training dataset", type=str, required=True)
parser.add_argument("--output_path", help="save path of pattern rate", type=str, default="patterns.txt")
args = parser.parse_args()

train_dataset_path = args.dataset_path
PCFG_rate_file = args.output_path

f_in = open(train_dataset_path, 'r', encoding='utf-8', errors='ignore')
f_out = open(PCFG_rate_file, 'w', encoding='utf-8', errors='ignore')

pcfg_patterns_dict = {}

lines = f_in.readlines()
total_num = len(lines)
for line in lines:
    if not line:
        continue
    password = line[:-1]
    pcfg_pattern = ' '.join(get_pattern(password))
    if pcfg_pattern in pcfg_patterns_dict:
        pcfg_patterns_dict[pcfg_pattern] += 1
    else:
        pcfg_patterns_dict[pcfg_pattern] = 1
    

pcfg_patterns_dict = dict(sorted(pcfg_patterns_dict.items(), key=lambda x:x[1], reverse=True))

for key,value in pcfg_patterns_dict.items():
    f_out.write(f'{key}\t{value/total_num}\n')