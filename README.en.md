Tiny Translator: A Minimal Transformer Translator (ZH→EN)

[![HF Model: caixiaoshun/tiny-translator-zh2en](https://img.shields.io/badge/HF%20Model-caixiaoshun%2Ftiny--translator--zh2en-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/caixiaoshun/tiny-translator-zh2en)
[![Spaces: Tiny-Translator](https://img.shields.io/badge/Spaces-Tiny--Translator-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/caixiaoshun/Tiny-Translator)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](#setup)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](#model-training)
[![Gradio 5.x](https://img.shields.io/badge/Gradio-5.x-FF7C00?logo=gradio&logoColor=white)](#web-demo-multiple-decoding-methods)
[![GitHub Stars](https://img.shields.io/github/stars/caixiaoshun/Tiny-Translator?style=social)](https://github.com/caixiaoshun/Tiny-Translator)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/caixiaoshun/Tiny-Translator/pulls)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Issues](https://img.shields.io/github/issues/caixiaoshun/Tiny-Translator.svg)](https://github.com/caixiaoshun/Tiny-Translator/issues)
[![Last Commit](https://img.shields.io/github/last-commit/caixiaoshun/Tiny-Translator.svg)](https://github.com/caixiaoshun/Tiny-Translator/commits/master)

English | [简体中文](README.md)

## 🌟 Overview

A tiny, from-scratch Transformer encoder–decoder for Chinese→English translation. Clean code, low dependencies, CLI and Gradio demo — great for learning and small-scale experiments.

## ✨ Features

- Pure PyTorch: full MHA, positional encoding, encoder/decoder
- BPE tokenizer via Hugging Face Tokenizers ([SOS]/[EOS]/[PAD]/[UNK])
- Training with Lightning 2.x + TorchMetrics
- Decoding: greedy, top-p (nucleus), beam search
- One-command Gradio web demo with local-first, hub fallback

## ⚙️ Setup

- Python ≥ 3.11 (recommended 3.11.13)
- PyTorch 2.x (install a build that matches your CUDA/CPU)

Install (prefer a fresh virtual env):

```bash
pip install -r requirements.txt
# If you need a custom PyTorch build, see https://pytorch.org for CUDA-specific commands.
```

## 📦 Data

- Expected path: `data/wmt_zh_en_training_corpus.csv`
- CSV must have a header with columns `0` (Chinese) and `1` (English), UTF-8
- Optional cache: enable `--use_cache` to create `data/cache.pickle`

Example download (ModelScope):

```bash
mkdir -p data
wget -O data/wmt_zh_en_training_corpus.csv \
  https://www.modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/resolve/master/wmt_zh_en_training_corpus.csv
```

## 🧩 Tokenizer Training

```bash
python -m src.train_tokenizer
```

This saves `checkpoints/tokenizer.json`. The tokenizer is trained jointly on ZH+EN text (see `src/train_tokenizer.py`).

## 🏋️ Model Training

Single-GPU example:

```bash
python -m src.train --use_cache --pin_memory --compile \
  --tokenizer_file checkpoints/tokenizer.json \
  --wmt_zh_en_path data/wmt_zh_en_training_corpus.csv
```

Outputs:

- TensorBoard logs: `log/tensorboard/runs/`
- Checkpoints: `log/checkpoint/translate-step=xxxxx.ckpt`

Tune hyperparams via `src/config.py` and `src/train.py` (e.g., `--embed_dim`, `--num_heads`, `--batch_size`, `--max_epochs`, `--vocab_size`, `--max_len`).

Tip: `--compile` requires PyTorch 2.x and a supported backend. If it fails, remove the flag.

## 🏅 Weights

- Hardware: 2× NVIDIA RTX 3090 (24GB each)
- Training script: `script/train.sh`
- Released checkpoint: `checkpoints/translate-step=290000.ckpt`

Note: During training, intermediate checkpoints are written to `log/checkpoint/`. For convenience, a 290000-step checkpoint is provided and used by default in the web demo and CLI examples.

## 🚀 Inference

### CLI (greedy decoding)

```bash
python -m src.sample --ckpt_path checkpoints/translate-step=290000.ckpt --zh "早上好"
```

Note: `src/sample.py` currently uses greedy decoding; top-p/beam are provided in the web demo.

### 🌐 Web Demo (multiple decoding methods)

```bash
python app.py
```

Open http://localhost:7860. The app loads from local `checkpoints/` first; if missing, it falls back to the Hugging Face Hub.

Environment variables (local-first; can override hub files):

- `HF_REPO_ID` (default `caixiaoshun/tiny-translator-zh2en`)
- `CKPT_FILE` (default `translate-step=290000.ckpt`)
- `TOKENIZER_FILE` (default `tokenizer.json`)
- `LOCAL_CKPT_PATH` (default `checkpoints/translate-step=290000.ckpt`)
- `LOCAL_TOKENIZER_PATH` (default `checkpoints/tokenizer.json`)
- `PORT` (default `7860`)

## 🧪 Scripts

- Training: `script/train.sh`

  ```bash
  bash script/train.sh
  ```

- Inference: `script/sample.sh`

  ```bash
  bash script/sample.sh
  ```

## 🗂️ Structure

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
  ├─ sample.py              # CLI inference (greedy)
  ├─ train.py               # Lightning training entry
  └─ train_tokenizer.py     # Train BPE tokenizer
data/
  └─ wmt_zh_en_training_corpus.csv
checkpoints/
  ├─ tokenizer.json
  └─ translate-step=290000.ckpt
log/
  ├─ checkpoint/            # ckpts produced by training (after running)
  └─ tensorboard/           # TensorBoard logs (after running)
```

## 🔗 Resources

- HF Model: https://huggingface.co/caixiaoshun/tiny-translator-zh2en
- HF Space: https://huggingface.co/spaces/caixiaoshun/Tiny-Translator

## 🛠️ Troubleshooting

- Missing tokenizer? Run `python -m src.train_tokenizer` or place `checkpoints/tokenizer.json`.
- Missing Lightning/TorchMetrics? `pip install -r requirements.txt` or install them separately.
- CSV errors? Ensure header with columns `0` and `1`, UTF-8 encoding.
- OOM? Reduce `--batch_size`, `--embed_dim`, `--num_heads`, or `--max_len`.
- `torch.compile` issues? Remove `--compile` if your backend doesn’t support it.
- Where are training ckpts? In `log/checkpoint/`. To use in the demo, copy desired weights to `checkpoints/`.

## 🤝 Contributing

Issues and PRs are welcome 🙌

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
