dataset_name="rockyou"
cleaned_dataset="./dataset/${dataset_name}-cleaned.txt"
model_path="./model/last-step/"
output_path="./generate/"

# 1. Get patterns rate
python get_pattern_rate.py --dataset_path=$cleaned_dataset
# 2. Generate (use DC-GEN) or Generate (not use DC-GEN)
python DC-GEN.py --model_path=$model_path --output_path=$output_path
# python normal-gen.py --model_path=$model_path --output_path=$output_path