import os
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanMetric, MaxMetric
import lightning as L
from torchmetrics.classification.accuracy import Accuracy
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tokenizers import Tokenizer
from src.config import Config
from src.model import TranslateModel
from src.dataset import TranslateDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint


import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--encoder_layer", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--decoder_layer", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--drop_out", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")

    parser.add_argument("--wmt_zh_en_path", type=str,
                        default="data/wmt_zh_en_training_corpus.csv",
                        help="Path to WMT zh-en training corpus")
    parser.add_argument("--tokenizer_file", type=str,
                        default="checkpoints/tokenizer.json",
                        help="Path to tokenizer file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory in dataloader")

    parser.add_argument("--tensorboard_dir", type=str, default="log/tensorboard",
                        help="Directory for tensorboard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="log/checkpoint",
                        help="Directory for saving checkpoints")

    # -------- optimizer / scheduler 参数 --------
    parser.add_argument("--base_lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.98), help="AdamW betas")
    parser.add_argument("--eps", type=float, default=1e-9, help="AdamW epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")

    parser.add_argument("--warmup_ratio", type=float, default=0.0005,
                        help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--start_factor", type=float, default=1e-3,
                        help="Linear warmup start factor (relative LR scale)")
    parser.add_argument("--end_factor", type=float, default=1.0,
                        help="Linear warmup end factor (relative LR scale)")
    parser.add_argument("--eta_min", type=float, default=3e-6,
                        help="Minimum LR in cosine annealing")

    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs to train")

    # -------- dataset cache 参数 --------
    parser.add_argument("--data_cache_dir", type=str, default="data/cache.pickle",
                        help="Path to cache file for dataset")
    parser.add_argument("--use_cache", action="store_true",
                        help="Enable dataset caching with pickle")
    
    parser.add_argument("--every_n_train_steps", type=int, default=10000,
                        help="Save checkpoint every N training steps")

    return parser.parse_args()



def merge_args_config(config: Config, args):
    for k, v in vars(args).items():
        setattr(config, k, v)
    return config


class TranslateLitModule(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        tokenizer: Tokenizer = Tokenizer.from_file(config.tokenizer_file)
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.net = TranslateModel(config=config)
        self.train_loss = MeanMetric()
        self.train_acc = Accuracy(
            task="multiclass", num_classes=config.vocab_size, ignore_index=self.pad_id
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        self.val_loss = MeanMetric()
        self.val_acc = Accuracy(
            task="multiclass", num_classes=config.vocab_size, ignore_index=self.pad_id
        )

        self.test_loss = MeanMetric()
        self.test_acc = Accuracy(
            task="multiclass", num_classes=config.vocab_size, ignore_index=self.pad_id
        )

        self.val_acc_best = MaxMetric()

        self.config = config

    def forward(self, batch) -> torch.Tensor:
        pred = self.net.forward(
            src=batch["src"],
            tgt=batch["tgt"],
            src_pad_mask=batch["src_pad_mask"],
            tgt_pad_mask=batch["tgt_pad_mask"],
        )
        return pred

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.train_acc.reset()

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        self.test_loss.reset()
        self.test_acc.reset()

    def model_step(self, batch):

        logits = self.forward(batch)

        B, L, C = logits.shape

        loss = self.criterion(logits.reshape(-1, C), batch["label"].reshape(-1))
        preds = torch.argmax(logits, dim=-1)
        return loss, preds.reshape(-1), batch["label"].reshape(-1)

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        if self.config.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.base_lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        
        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = max(1, int(self.config.warmup_ratio * total_steps))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=self.config.start_factor,
            end_factor=self.config.end_factor,
            total_iters=warmup_steps,
        )


        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.config.eta_min,
        )

        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # 每个 step 调整
                "frequency": 1
            }
        }


def prepare_dataloader(dataset, config: Config, shuffle=True):
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return dataloader


def prepare_dataset(config: Config):
    full_ds = TranslateDataset(config=config)
    val_ratio = config.val_ratio
    val_len = int(len(full_ds) * val_ratio)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(
        full_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config.seed),
    )
    return prepare_dataloader(train_ds, config), prepare_dataloader(
        val_ds, config, False
    )


def prepare_callback(config: Config):
    logger = TensorBoardLogger(save_dir=config.tensorboard_dir, name="runs")
    rich_progress_bar = RichProgressBar()
    checkpoint = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="translate-{step:05d}",
        save_weights_only=True,
        every_n_train_steps=config.every_n_train_steps,
        save_top_k=-1,
    )
    return logger, [rich_progress_bar, checkpoint]


def main():

    args = parser_args()
    config = merge_args_config(Config(), args)

    L.seed_everything(config.seed)

    train_loader, val_loader = prepare_dataset(config)

    model = TranslateLitModule(config=config)

    logger, callbacks = prepare_callback(config)

    trainer = L.Trainer(callbacks=callbacks, logger=logger, max_epochs=config.max_epochs)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
