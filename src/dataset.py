from torch.utils.data import Dataset
import csv
from tokenizers import Tokenizer
import torch
import os
import pickle
from src.config import Config


class TranslateDataset(Dataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokenizer: Tokenizer = Tokenizer.from_file(config.tokenizer_file)
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.pairs = []
        if os.path.exists(config.data_cache_dir) and config.use_cache:
            with open(config.data_cache_dir, "rb") as f:
                self.pairs = pickle.load(f)
        else:
            with open(self.config.wmt_zh_en_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for line in reader:
                    self.pairs.append((line["0"], line["1"]))
            if config.use_cache:
                with open(config.data_cache_dir, "wb") as cache_f:
                    pickle.dump(self.pairs, cache_f)

    def __len__(self):
        return len(self.pairs)

    def encode(self, text):
        ids = self.tokenizer.encode(text).ids

        if len(ids) > self.config.max_len:
            ids = ids[: self.config.max_len]

        pad_len = self.config.max_len - len(ids)

        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
        pad_mask = [False if i == self.pad_id else True for i in ids]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(
            pad_mask, dtype=torch.bool
        )

    def __getitem__(self, idx):
        zh, en = self.pairs[idx]

        zh_id, zh_pad = self.encode(zh)

        en_id, en_pad = self.encode(en)

        return dict(
            src=zh_id,
            src_pad_mask=zh_pad,
            tgt=en_id[:-1],
            tgt_pad_mask=en_pad[:-1],
            label=en_id[1:],
        )
