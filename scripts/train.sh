dataset_name="rockyou"
ready4train_dataset="./dataset/${dataset_name}-cleaned-Train-ready.txt"

# 1. generate vocab file
python generate_vocab_file.py
# 2. train
python train.py --dataset_path=$ready4train_dataset