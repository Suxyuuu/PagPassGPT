dataset_name="rockyou"
original_dataset="./dataset/${dataset_name}.txt"
cleaned_dataset="./dataset/${dataset_name}-cleaned.txt"
training_dataset="./dataset/${dataset_name}-cleaned-Train.txt"
test_dataset="./dataset/${dataset_name}-cleaned-Test.txt"
ready4train_dataset="./dataset/${dataset_name}-cleaned-Train-ready.txt"

# 1. clean dataset
python clean_dataset.py --dataset_path=$original_dataset --output_path=$cleaned_dataset
# 2. split into training set and test set
python split_dataset.py --dataset_path=$cleaned_dataset --train_path=$training_dataset --test_path=$test_dataset
# 3. concat pattern and password together
python concat_pattern_password.py --dataset_path=$training_dataset --output_path=$ready4train_dataset