# This file aims to train a PagPassGPT.

from tokenizer.char_tokenizer import CharTokenizer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import time
import argparse

parser = argparse.ArgumentParser()
# file path parameter setting
parser.add_argument("--dataset_path", help="path of preprocessed train dataset", type=str, required=True)
parser.add_argument("--vocabfile_path", help="path of vocab file", type=str, default="./tokenizer/vocab.json")
parser.add_argument("--model_path", help="directory to save model", type=str, default="./model/")
parser.add_argument("--log_path", help="directory of log", type=str, default="./log/")
# environment parameter setting
parser.add_argument("--random_seed", help="random seed", type=int, default=42)
parser.add_argument("--num_processer", help="num of processer (cpu logit cores)", type=int, default=1)
# model parameter setting
parser.add_argument("--input_size", help="should be larger than (2*max len of password + 3), default is 32 according to max_len=12", type=int, default=32)
parser.add_argument("--embed_size", help="embedding size", type=int, default=384)
parser.add_argument("--layer_num", help="num of layers", type=int, default=8)
parser.add_argument("--head_num", help="num of multi head", type=int, default=12)
# training parameter setting
parser.add_argument("--epoch_num", help="num of epoch (containing early stop))", type=int, default=30)
parser.add_argument("--batch_size", help="batch_size", type=int, default=512)
parser.add_argument("--eval_step", help="eval model every n steps", type=int, default=2000)
parser.add_argument("--save_step", help="save model every n steps", type=int, default=6000)
parser.add_argument("--early_stop", help="early stop patience", type=int, default=3)

args = parser.parse_args()

train_dataset_path = args.dataset_path
vocab_file = args.vocabfile_path
model_output_dir = args.model_path
log_dir = args.log_path

random_seed = args.random_seed
num_processer = args.num_processer

# model params: 14260608
input_size = args.input_size
embed_size = args.embed_size
layer_num = args.layer_num
head_num = args.head_num

epoch_num = args.epoch_num
batch_size = args.batch_size
eval_step = args.eval_step
save_step = args.save_step
early_stop = args.early_stop

print(f'Load tokenizer.')
tokenizer = CharTokenizer(vocab_file=vocab_file, 
                          bos_token="<BOS>",
                          eos_token="<EOS>",
                          sep_token="<SEP>",
                          unk_token="<UNK>",
                          pad_token="<PAD>",
                          )


print(f'Load dataset.')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataset = load_dataset('text', data_files=train_dataset_path, num_proc=num_processer, split='train')
train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], max_len=input_size, padding=True), batched=True)

print(f'Split dataset into training set and validation set.')
train_dataset= train_dataset.train_test_split(test_size=0.125)
eval_dataset = train_dataset['test']
train_dataset = train_dataset['train']

print(f'Load model config.')
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=input_size,
    n_embd=embed_size,
    n_layer=layer_num,
    n_head=head_num,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    scale_attn_by_inverse_layer_idx=False,
    reorder_and_upcast_attn=False,
)

model = GPT2LMHeadModel(config=config)
print(f"Num parameters: {model.num_parameters()}")
print(model)


print(f'Load training config.')
training_args = TrainingArguments(
    output_dir=model_output_dir, 
    overwrite_output_dir=True, 
    num_train_epochs=epoch_num,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_steps = eval_step,
    save_steps=save_step,
    save_strategy='steps',
    evaluation_strategy='steps',
    prediction_loss_only=True,
    logging_dir=log_dir + time.strftime("%Y%m%d-%H:%M", time.localtime()),
    seed=random_seed,
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
    save_total_limit=1
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop)],
    
)

print(f'*'*30)
print(f'Training begin.')
trainer.train()

trainer.save_model(model_output_dir+"last-step/")
print(f'Model saved in {model_output_dir+"last-step/"}')
print(f'*'*30)
print(f'Training done.')