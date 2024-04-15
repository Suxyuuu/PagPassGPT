"""
This file aims to process the input passwords to the rule as follows for convience to train:
    
    pattern <SEP> password

"""


import argparse


def get_pattern(password:str):
    result = []
    
    current_type = None
    current_length = 0
    
    for char in password:
        if char.isalpha():
            if current_type == 'L':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'L'
                current_length = 1
        elif char.isdigit():
            if current_type == 'N':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'N'
                current_length = 1
        else:
            if current_type == 'S':
                current_length += 1
            else:
                if current_type:
                    result.append(current_type + str(current_length))
                current_type = 'S'
                current_length = 1
    
    if current_type:
        result.append(current_type + str(current_length))
    return result


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="path of training dataset after split", type=str, required=True)
    parser.add_argument("--output_path", help="path of output dataset (ready for training)", type=str, required=True)
    args = parser.parse_args()
    
    input_dataset = args.dataset_path
    output_dataset = args.output_path
    f_in = open(input_dataset, 'r', encoding='utf-8', errors='ignore')
    f_out = open(output_dataset, 'w', encoding='utf-8', errors='ignore')

    lines = f_in.readlines()

    for line in lines:
        password = line[:-1]
        prompt = ' '.join(get_pattern(password))
        new_line = prompt + ' <SEP> ' + ' '.join(list(password)) + '\n'
        f_out.write(new_line)
