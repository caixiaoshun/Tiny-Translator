Tiny Translator: A Minimal Transformer Translator (ZH→EN)

[![HF Model: caixiaoshun/tiny-translator-zh2en](https://img.shields.io/badge/HF%20Model-caixiaoshun%2Ftiny--translator--zh2en-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/caixiaoshun/tiny-translator-zh2en)
[![Spaces: Tiny-Translator](https://img.shields.io/badge/Spaces-Tiny--Translator-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/caixiaoshun/Tiny-Translator)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](#setup)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](#training)
[![GitHub Stars](https://img.shields.io/github/stars/caixiaoshun/Tiny-Translator?style=social)](https://github.com/caixiaoshun/Tiny-Translator)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/caixiaoshun/Tiny-Translator/pulls)

English | [简体中文](README.md)

## Overview

A tiny, from-scratch Transformer encoder–decoder for Chinese→English translation. Clean code, low deps, CLI and Gradio demo — great for learning and small experiments.

## Features

- Pure PyTorch (multi-head attention, positional encoding, encoder/decoder)
- BPE tokenizer via Hugging Face Tokenizers ([SOS]/[EOS]/[PAD]/[UNK])
- Lightning 2.x training with TorchMetrics
- Decoding: greedy, top-p (nucleus), beam search
- One-command Gradio web demo

## Quick Start

### Setup

Requirement: Python 3.11.13

```bash
pip install -r requirements.txt
pip install tokenizers lightning torchmetrics
```

### Run Web Demo

```bash
python app.py
```

Open http://localhost:7860. The app prefers local checkpoints/tokens in `checkpoints/` and falls back to the Hugging Face Hub. Behavior can be tuned by env vars in `app.py`.

### CLI Inference

```bash
python -m src.sample --ckpt_path checkpoints/translate-step=290000.ckpt --zh "早上好"
```

## Data

- Expected path: `data/wmt_zh_en_training_corpus.csv`
- CSV must contain a header with columns `0` (Chinese) and `1` (English)
- Optional cache with `--use_cache` to `data/cache.pickle`

Download example from ModelScope:

```bash
mkdir -p data
wget -O data/wmt_zh_en_training_corpus.csv \
  https://www.modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/resolve/master/wmt_zh_en_training_corpus.csv
```

## Training

Train tokenizer:

```bash
python -m src.train_tokenizer
```

Start training (single GPU example):

```bash
python -m src.train --use_cache --pin_memory --compile \
  --tokenizer_file checkpoints/tokenizer.json \
  --wmt_zh_en_path data/wmt_zh_en_training_corpus.csv
```

See `src/config.py` and `src/train.py` for tunables (e.g., `--embed_dim`, `--num_heads`, `--batch_size`, `--max_epochs`).

## Structure

```
app.py                      # Gradio web app
requirements.txt            # Dependencies
script/
  ├─ train.sh               # Training example
  └─ sample.sh              # Inference example
src/
  ├─ config.py              # Defaults
  ├─ dataset.py             # Data loading & cache
  ├─ model.py               # Transformer implementation
  ├─ sample.py              # CLI inference
  ├─ train.py               # Lightning training entry
  └─ train_tokenizer.py     # Train BPE tokenizer
data/
  └─ wmt_zh_en_training_corpus.csv
checkpoints/
  ├─ tokenizer.json
  └─ translate-step=290000.ckpt
```

## Resources

- HF Model: https://huggingface.co/caixiaoshun/tiny-translator-zh2en
- HF Space: https://huggingface.co/spaces/caixiaoshun/Tiny-Translator

Env vars (local-first, hub fallback/override): `HF_REPO_ID`, `CKPT_FILE`, `TOKENIZER_FILE`, `LOCAL_CKPT_PATH`, `LOCAL_TOKENIZER_PATH`, `PORT`.

## Troubleshooting

- Missing tokenizer: install `tokenizers` or run `python -m src.train_tokenizer`
- Missing Lightning/TorchMetrics: `pip install lightning torchmetrics`
- CSV errors: ensure header with columns `0` and `1` (UTF-8)
- OOM: reduce `--batch_size`, `--embed_dim`, `--num_heads`, or `--max_len`

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
