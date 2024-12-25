# PagPassGPT

Codes for Paper: [PagPassGPT: Pattern Guided Password Guessing via Generative Pretrained Transformer
](https://www.computer.org/csdl/proceedings-article/dsn/2024/410500a429/1ZPxTMt2ao8) in DSN 2024.

## 1 Environment

```shell
conda create -n env_name python=3.8.10
conda activate env_name
pip install -r requirements.txt
# pip install numpy==1.24.2 huggingface-hub==0.13.4 fsspec==2022.11.0 torch==2.0.0 transformers==4.29.0 datasets==2.12.0 accelerate==0.17.1
```


## 2 Usage

### 2.1 Prepare datasets

1. You should have a dataset of passwords, like **"RockYou"** or other datasets. And you should make sure the dataset contains only passwords.

2. Run the script `preprocess.sh` to preprocess datasets.
```shell
sh ./scripts/preprocess.sh
```

*Note: Here gives the **"RockYou"** dataset [download link](https://www.google.com/url?sa=i&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&psig=AOvVaw3rovncwk_ZO-AVgMK56N5-&ust=1734701481601000&source=images&cd=vfe&opi=89978449&ved=0CAYQrpoMahcKEwiwiazf-LOKAxUAAAAAHQAAAAAQBA).*


### 2.2 Train a PagPassGPT

Run the script `train.sh` to train.
```shell
sh ./scripts/train.sh
```

### 2.3 Generate passwords
Run the script `generate.sh` to generate.

```shell
sh ./scripts/generate.sh
```

*Note: In this shell, you can choose to use **D\&C-GEN** or not by changing just one line.*

### 2.4 Evaluate passwords

```shell
sh ./scripts/evaluate.sh
```
*Note: The evaluation mainly focus on **Hit rate** and **Repeat rate**.*

## 3 Update Logs

+ 2024.12.25: Fix some bugs.
+ 2024.12.20: Add pip requirements.
+ 2024.12.19: Update all codes.
  + Fix some bugs.
  + Provide more precise environmental requirements.
  + Provide new files for evaluation.
  + Make codes more user-friendly. 
  + Update `README.md`.
+ 2024.12.12: Update the paper link (from arXiv to IEEE).
+ 2024.4.15: Upload the codes firstly.
