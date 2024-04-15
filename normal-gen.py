from transformers import (
    GPT2LMHeadModel
)
import time
import threading
import torch
from tokenizer import CharTokenizer
import argparse


MAX_LEN = 32    # It should be equal to input size of model.

class ThreadBase(threading.Thread):
    """ overload threading, so that it can return values """
    def __init__(self, target=None, args=()):
        super().__init__()
        self.func = target
        self.args = args
 
    def run(self):
        self.result = self.func(*self.args)
 
    def get_result(self):
        try:
            return self.result
        except Exception as e:
            print(e)
            return None

def gen_sample(test_model_path, tokenizer, GEN_BATCH_SIZE, GPU_ID):
    model = GPT2LMHeadModel.from_pretrained(test_model_path)
    
    device = "cuda:"+str(GPU_ID)
    model.to(device)

    inputs = ""
    tokenizer_forgen_result = tokenizer.encode_forgen(inputs)
    passwords = set()
    
    outputs = model.generate(
        input_ids= tokenizer_forgen_result.view([1,-1]).to(device),
        pad_token_id=tokenizer.pad_token_id,
        max_length=MAX_LEN, 
        do_sample=True, 
        num_return_sequences=GEN_BATCH_SIZE,
        )
    outputs = tokenizer.batch_decode(outputs)
    for output in outputs:
        passwords.add(output)

    return [*passwords,]


def gen_parallel(vocab_file, batch_size, test_model_path, N, gen_passwords_path):
    print(f'Load tokenizer.')
    tokenizer = CharTokenizer(vocab_file=vocab_file, 
                              bos_token="<BOS>",
                              eos_token="<EOS>",
                              sep_token="<SEP>",
                              unk_token="<UNK>",
                              pad_token="<PAD>"
                              )
    tokenizer.padding_side = "left"

    # mulit gpu parallel
    if not torch.cuda.is_available():
        print('ERROR! GPU not found!')
    else:
        num_gpus = torch.cuda.device_count()
        total_start = time.time()
        threads = {}
        total_passwords = []

        total_round = N//batch_size
        print('*'*30)
        print(f'Generation begin.')
        print('Total generation needs {} batchs.'.format(total_round))

        i = 0
        gpu_id = 0
        while(i < total_round or len(threads) > 0 ):
            if len(threads) == 0:
                for gpu_id in range(num_gpus):
                    if i < total_round:
                        t=ThreadBase(target=gen_sample, args=(test_model_path, tokenizer, batch_size, gpu_id))
                        t.start()
                        threads[t] = i
                        i += 1
            
            # check whether some threads have finished.
            temp_threads = threads.copy()
            for t in temp_threads:
                t.join()
                if not t.is_alive():
                    new_passwords = t.get_result()
                    new_num = len(new_passwords)
                    total_passwords += new_passwords
                    print('[{}/{}] generated {}.'.format(temp_threads[t]+1, total_round, new_num))
                    threads.pop(t)
               
        total_passwords = set(total_passwords)

        gen_passwords_path = gen_passwords_path + f'{n}-normal' + '.txt'
        
        f_gen = open(gen_passwords_path, 'w', encoding='utf-8', errors='ignore')
        for password in total_passwords:
            f_gen.write(password+'\n')

        total_end = time.time()
        total_time = total_end-total_start
        
        print('Generation file saved in: {}'.format(gen_passwords_path))
        print('Generation done.')
        print('*'*30)
        print('Use time:{}'.format(total_time))
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="directory of pagpassgpt", type=str, required=True)
    parser.add_argument("--vocabfile_path", help="path of vocab file", type=str, default='./tokenizer/vocab.json')
    parser.add_argument("--output_path", help="path of output file path", type=str, required=True)
    parser.add_argument("--generate_num", help="total guessing number", default=1000000, type=int)
    parser.add_argument("--batch_size", help="generate batch size", default=5000, type=int)
    args = parser.parse_args()

    model_path = args.model_path
    vocab_file = args.vocabfile_path
    output_path = args.output_path

    n = args.generate_num
    batch_size = args.batch_size
    
    gen_parallel(vocab_file, batch_size, model_path, n, output_path)