"""
This file aims to split whole dataset into training set and test set:
    
    Training set will be used in training and will be split again (for validation).
    Test set will be used in evaluation.

"""

import random
import argparse

def split_train_test(file_path, ratio, train_path, test_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print('Shuffling passwords.')
    random.shuffle(lines)
    f.close()

    split = int(len(lines) * ratio)

    with open(train_path, 'w') as f:
        print('Saving 80% ({}) of dataset for training in {}'.format(split, train_path))
        f.write(''.join(lines[0:split]))
    f.close()

    with open(test_path, 'w') as f:
        print('Saving 20% ({}) of dataset for test in {}'.format(len(lines) - split, test_path))
        f.write(''.join(lines[split:]))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="path of cleaned dataset", type=str, required=True)
    parser.add_argument("--train_path", help="save path of training set after split", type=str, required=True)
    parser.add_argument("--test_path", help="save path of test set after split", type=str, required=True)
    parser.add_argument("--ratio", help="split ratio", type=float, default=0.8)
    args = parser.parse_args()

    print(f'Split begin.')
    split_train_test(args.dataset_path, args.ratio, args.train_path, args.test_path)
    print(f'Split done.')