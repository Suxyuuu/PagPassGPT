# This file aims to generate vocab file with special pattern tokens.

import json
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", help="save path of vocab file", type=str, default='./tokenizer/vocab.json')
parser.add_argument("--max_len", help="max length of password in datasets", default=12, type=int)
args = parser.parse_args()

# ASCII
Number = [(48, 57)]
Letter = [(65, 90), (97, 122)]
Special_char = [(33, 47), (58, 64), (91, 96), (123, 126)]

chars = [Number, Letter, Special_char]

Special_token = ["<BOS>", "<SEP>", "<EOS>", "<UNK>", "<PAD>"]

vocab_dict = OrderedDict()
index = 0

for token in Special_token:
    vocab_dict[token] = index
    index += 1

for char_type in ['N', 'L', 'S']:
    for length in range(args.max_len, 0, -1):
        vocab_dict[char_type+str(length)] = index
        index += 1

for char_type in chars:
    for turple in char_type:
        for i in range(turple[0], turple[1]+1):
            vocab_dict[chr(i)] = index
            index += 1

json_str = json.dumps(vocab_dict, indent=4)
with open(args.save_path, 'w') as json_file:
    json_file.write(json_str)


