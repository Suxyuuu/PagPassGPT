dataset_name="rockyou"
test_dataset="./dataset/${dataset_name}-cleaned-Test.txt"
# output_path should change the generate num by the generate_num in generate.sh
output_path="./generate/1000000/"

# 1. evaluate generated passwords in the normal method
python evaluate.py --test_file="$test_dataset" --gen_path="$gen_path" --isNormal
# 2. evaluate generated passwords in the DC-GEN method
python evaluate.py --test_file="$test_dataset" --gen_path="$gen_path"