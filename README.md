# PagPassGPT
PagPassGPT: Pattern Guided Password Guessing via Generative Pretrained Transformer

Paper: https://arxiv.org/abs/2404.04886v1

## 1 Environment

```shell
pip install torch transformers datasets
```

## 2 How to use

### 2.1 Train a PagPassGPT

Firstly, you should have a dataset of passwords, like `rockyou` or other datasets. And you should make sure the dataset contains only passwords.

Next, run the script to preprocess datasets.
```shell
sh ./scripts/preprocess.sh
```

Last, run the script to train.
```shell
sh ./scripts/train.sh
```

### 2.2 Generate passwords

```shell
sh ./scripts/generate.sh
```

In this shell, you can choose to use DC-GEN or not by changing just one line.
