<h1 align="center">Tiny Translator</h1>
<p align="center">极简 Transformer 中文 → 英文翻译器</p>

<p align="center">
  <a href="https://huggingface.co/caixiaoshun/tiny-translator-zh2en">
    <img alt="HF Model: caixiaoshun/tiny-translator-zh2en" src="https://img.shields.io/badge/HF%20Model-caixiaoshun%2Ftiny--translator--zh2en-FFD21E?logo=huggingface" />
  </a>
  <a href="https://huggingface.co/spaces/caixiaoshun/Tiny-Translator">
    <img alt="Spaces: Tiny-Translator" src="https://img.shields.io/badge/Spaces-Tiny--Translator-FFD21E?logo=gradio" />
  </a>
  <a href="#%E7%8E%AF%E5%A2%83%E4%B8%8E%E5%AE%89%E8%A3%85">
    <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python" />
  </a>
  <a href="#%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83">
    <img alt="PyTorch 2.x" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" />
  </a>
  <a href="#%E7%BD%91%E9%A1%B5-demo%E6%94%AF%E6%8C%81%E5%A4%9A%E7%A7%8D%E8%A7%A3%E7%A0%81">
    <img alt="Gradio 5.x" src="https://img.shields.io/badge/Gradio-5.x-FF7C00?logo=gradio" />
  </a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" />
  </a>
  <br/>
  简体中文 | <a href="README.en.md">English</a>
</p>

<br/>

## 🌟 简介

一个从零实现的极简 Transformer 编码器-解码器，用于中文→英文翻译。代码清晰、依赖少，支持命令行和 Gradio 网页 Demo，适合学习与小规模实验。

## ✨ 主要特性

- 纯 PyTorch：多头注意力、位置编码、Encoder/Decoder 全量实现
- BPE 分词器：基于 Hugging Face Tokenizers 训练（含 [SOS]/[EOS]/[PAD]/[UNK]）
- 训练框架：Lightning 2.x + TorchMetrics 指标
- 解码方式：Greedy、Top-p（核采样）、Beam Search
- 一键网页 Demo：本地优先，缺失则自动回落 Hugging Face Hub

## ⚙️ 环境与安装

- Python ≥ 3.11（推荐 3.11.13）
- PyTorch 2.x（请安装与你 CUDA/CPU 匹配的版本）

快速安装（建议在全新虚拟环境中执行）：

```bash
pip install -r requirements.txt
# 若你需要自定义 PyTorch 构建，请参考 https://pytorch.org 获取与你 CUDA 版本匹配的安装命令。
```

可选：使用 Conda/Mamba 创建环境以减少依赖冲突。

## 📦 数据准备

- 期望路径：`data/wmt_zh_en_training_corpus.csv`
- CSV 必须包含表头，列名为 `0`（中文）与 `1`（英文），编码 UTF-8
- 可选缓存：训练时加 `--use_cache` 将生成 `data/cache.pickle`

示例（来自 ModelScope）：

```bash
mkdir -p data
wget -O data/wmt_zh_en_training_corpus.csv \
  https://www.modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/resolve/master/wmt_zh_en_training_corpus.csv
```

## 🧩 分词器训练

```bash
python -m src.train_tokenizer
```

默认会将分词器保存到 `checkpoints/tokenizer.json`。分词训练会将中英文拼接后共同学习 BPE 词表（见 `src/train_tokenizer.py`）。

## 🏋️ 模型训练

单卡示例：

```bash
python -m src.train --use_cache --pin_memory --compile \
  --tokenizer_file checkpoints/tokenizer.json \
  --wmt_zh_en_path data/wmt_zh_en_training_corpus.csv
```

输出目录：

- TensorBoard 日志：`log/tensorboard/runs/`
- 训练检查点：`log/checkpoint/translate-step=xxxxx.ckpt`

如需调整结构/超参，请参考 `src/config.py` 与 `src/train.py`（例如：`--embed_dim`、`--num_heads`、`--batch_size`、`--max_epochs`、`--vocab_size`、`--max_len` 等）。

提示：`--compile` 需要 PyTorch 2.x 且后端支持，若报错可去掉该参数。

## 🏅 权重

- 训练硬件：2× NVIDIA RTX 3090（每张 24GB 显存）
- 训练脚本：`script/train.sh`
- 已发布权重：`checkpoints/translate-step=290000.ckpt`

说明：训练过程中会将中间权重保存到 `log/checkpoint/`，为了便于体验，我们同时在仓库中提供了 290000 step 的检查点，网页 Demo 与 CLI 示例默认使用该权重。

## 🚀 推理

### 命令行（贪心解码）

```bash
python -m src.sample --ckpt_path checkpoints/translate-step=290000.ckpt --zh "早上好"
```

注意：`src/sample.py` 当前使用贪心解码；Top-p/Beam Search 已在网页 Demo 中提供。

### 🌐 网页 Demo（支持多种解码）

```bash
python app.py
```

启动后访问 http://localhost:7860 。应用会优先加载本地 `checkpoints/` 下的 `tokenizer.json` 与权重；若缺失将自动从 Hugging Face Hub 拉取（见下方环境变量）。

可用环境变量（本地优先，支持覆盖 Hub 文件名）：

- `HF_REPO_ID`（默认 `caixiaoshun/tiny-translator-zh2en`）
- `CKPT_FILE`（默认 `translate-step=290000.ckpt`）
- `TOKENIZER_FILE`（默认 `tokenizer.json`）
- `LOCAL_CKPT_PATH`（默认 `checkpoints/translate-step=290000.ckpt`）
- `LOCAL_TOKENIZER_PATH`（默认 `checkpoints/tokenizer.json`）
- `PORT`（默认 `7860`）

## 🧪 脚本

- 训练：`script/train.sh`

  ```bash
  bash script/train.sh
  ```

- 推理：`script/sample.sh`

  ```bash
  bash script/sample.sh
  ```

## 🗂️ 目录结构

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
  ├─ sample.py              # 命令行推理（贪心）
  ├─ train.py               # Lightning 训练入口
  └─ train_tokenizer.py     # 训练 BPE 分词器
data/
  └─ wmt_zh_en_training_corpus.csv
checkpoints/
  ├─ tokenizer.json
  └─ translate-step=290000.ckpt
log/
  ├─ checkpoint/            # 训练输出的 ckpt（运行后生成）
  └─ tensorboard/           # TensorBoard 日志（运行后生成）
```

## 🔗 模型与资源

- Hugging Face 模型仓库：
  - https://huggingface.co/caixiaoshun/tiny-translator-zh2en
- 在线体验（Spaces）：
  - https://huggingface.co/spaces/caixiaoshun/Tiny-Translator

## 🛠️ 常见问题（FAQ）

- 分词器文件缺失？先执行 `python -m src.train_tokenizer` 或放置到 `checkpoints/tokenizer.json`。
- 无法导入 Lightning/TorchMetrics？执行 `pip install -r requirements.txt`，或单独安装 `lightning torchmetrics`。
- CSV 读取报错？确保存在表头，列名为 `0` 与 `1`，并使用 UTF-8 编码。
- 显存不足（OOM）？尝试减小 `--batch_size`、`--embed_dim`、`--num_heads` 或 `--max_len`。
- `torch.compile` 报错？与硬件/后端相关，可先移除 `--compile`。
- 训练出的 ckpt 在哪里？默认在 `log/checkpoint/`，如要在 Demo 中使用，可将需要的权重复制到 `checkpoints/`。

## 🤝 贡献

欢迎 Issue/PR 指出问题或改进建议 🙌

## 📄 许可证

本项目采用 Apache License 2.0，详见 [LICENSE](LICENSE)。
