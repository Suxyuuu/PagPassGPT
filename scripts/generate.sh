dataset_name="rockyou"
cleaned_dataset="./dataset/${dataset_name}-cleaned.txt"
model_path="./model/last-step/"
output_path="./generate/"

# 1. Get patterns rate
python get_pattern_rate.py --dataset_path=$cleaned_dataset
# 2. Generate (use DC-GEN) by using no.4 and no.5 gpus (ids of gpus should be continuous)
# python DC-GEN.py --model_path=$model_path --output_path=$output_path --generate_num=1000000 --batch_size=5000 --gpu_num=2 --gpu_index=4
# 3. or Generate (not use DC-GEN) 
python normal-gen.py --model_path=$model_path --output_path=$output_path --generate_num=1000000 --batch_size=5000 --gpu_num=2 --gpu_index=4