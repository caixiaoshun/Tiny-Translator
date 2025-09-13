Tiny Translator：极简 Transformer 中文→英文翻译器

[![HF Model: caixiaoshun/tiny-translator-zh2en](https://img.shields.io/badge/HF%20Model-caixiaoshun%2Ftiny--translator--zh2en-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/caixiaoshun/tiny-translator-zh2en)
[![Spaces: Tiny-Translator](https://img.shields.io/badge/Spaces-Tiny--Translator-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/caixiaoshun/Tiny-Translator)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](#%E5%AE%89%E8%A3%85)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](#%E8%AE%AD%E7%BB%83)
[![GitHub Stars](https://img.shields.io/github/stars/caixiaoshun/Tiny-Translator?style=social)](https://github.com/caixiaoshun/Tiny-Translator)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/caixiaoshun/Tiny-Translator/pulls)

简体中文 | [English](README.en.md)

## 简介

一个从零实现的极简 Transformer 编码器-解码器，用于中文→英文翻译。代码清晰、依赖少，支持命令行与 Gradio 网页演示，适合学习与小规模实验。

## 主要特性

- 纯 PyTorch 实现（多头注意力、位置编码、编码器/解码器）
- Hugging Face Tokenizers 训练 BPE 分词器（含 [SOS]/[EOS]/[PAD]/[UNK]）
- Lightning 2.x 训练流程，TorchMetrics 指标
- 提供贪心、Top-p（核采样）与 Beam Search 多种解码
- 一键启动 Gradio 网页 Demo

## 快速开始

### 安装

环境要求：Python 3.11.13

```bash
pip install -r requirements.txt
pip install tokenizers lightning torchmetrics
```

建议使用 Conda/Mamba 新建环境以减少依赖冲突。

### 运行网页 Demo

```bash
python app.py
```

浏览器打开 http://localhost:7860 即可。默认优先使用本地 `checkpoints/`，若缺失将回落到 Hugging Face Hub（可在 `app.py` 中通过环境变量控制）。

### 命令行推理

```bash
python -m src.sample --ckpt_path checkpoints/translate-step=290000.ckpt --zh "早上好"
```

## 数据

- 期望路径：`data/wmt_zh_en_training_corpus.csv`
- CSV 需包含表头，列名为 `0`（中文）与 `1`（英文）
- 可选缓存：加 `--use_cache` 后会写入 `data/cache.pickle`

从 ModelScope 下载示例：

```bash
mkdir -p data
wget -O data/wmt_zh_en_training_corpus.csv \
  https://www.modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/resolve/master/wmt_zh_en_training_corpus.csv
```

## 训练

训练分词器：

```bash
python -m src.train_tokenizer
```

启动训练（单卡示例）：

```bash
python -m src.train --use_cache --pin_memory --compile \
  --tokenizer_file checkpoints/tokenizer.json \
  --wmt_zh_en_path data/wmt_zh_en_training_corpus.csv
```

常用可调项见 `src/config.py` 与 `src/train.py` 的 CLI 参数（如 `--embed_dim`、`--num_heads`、`--batch_size`、`--max_epochs` 等）。

## 目录结构

```
app.py                      # Gradio 网页应用
requirements.txt            # 依赖
script/
  ├─ train.sh               # 训练示例脚本
  └─ sample.sh              # 推理示例脚本
src/
  ├─ config.py              # 默认配置
  ├─ dataset.py             # 数据加载与缓存
  ├─ model.py               # Transformer 实现
  ├─ sample.py              # 命令行推理
  ├─ train.py               # Lightning 训练入口
  └─ train_tokenizer.py     # 训练 BPE 分词器
data/
  └─ wmt_zh_en_training_corpus.csv
checkpoints/
  ├─ tokenizer.json
  └─ translate-step=290000.ckpt
```

## 模型与资源

- Hugging Face 模型仓库：
  - https://huggingface.co/caixiaoshun/tiny-translator-zh2en
- 在线体验（Spaces）：
  - https://huggingface.co/spaces/caixiaoshun/Tiny-Translator

环境变量（用于 Hub 回落/覆盖，本地优先）：`HF_REPO_ID`、`CKPT_FILE`、`TOKENIZER_FILE`、`LOCAL_CKPT_PATH`、`LOCAL_TOKENIZER_PATH`、`PORT`。

## 常见问题

- 找不到分词器：安装 `tokenizers`，或先执行 `python -m src.train_tokenizer`
- 缺少 Lightning/TorchMetrics：`pip install lightning torchmetrics`
- CSV 读取错误：确认表头和列名（`0` 与 `1`），编码为 UTF-8
- 显存不足：减小 `--batch_size`、`--embed_dim`、`--num_heads` 或 `--max_len`

## 许可

本项目采用 Apache License 2.0，详见 [LICENSE](LICENSE)。
