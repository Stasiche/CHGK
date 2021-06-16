import os
import hydra
import pytorch_lightning as pl
import torch
import numpy as np

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional, Tuple, Iterable, List, Union
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf

from src.training.models.GPT2SberSmall import GPT2SberSmall
from src.training.utils.chgk_datasets import GPT2SmallDataset
from definitions import ROOT_PATH


class GPT2Sber(pl.LightningModule):
    def __init__(
        self,
        model_path: str,
        lr: float,
        w_decay: float,
        warmup_steps: Union[int, float],
        eps: float,
        freeze_model: bool = False,
        scheduler: str = "const"
    ):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

        self.warmup_steps = warmup_steps

        self.train_TP_top1 = 0
        self.train_TP_topk = 0
        self.val_TP_top1 = 0
        self.val_TP_topk = 0

        self.train_samples = 0
        self.val_samples = 0

        self.train_loss = 0

        self.lr = lr
        self.w_decay = w_decay
        self.eps = eps
        self.freeze_model = freeze_model
        self.scheduler = scheduler

        self.save_hyperparameters()

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        self.model.eval()
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        # masking pad tokens not to calculate gradients on them
        labels[labels == 0] = -100

        # forward
        res = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # calculating metrics
        acc_top1, acc_topk = self._calc_acc(res.logits, labels, True)
        self.train_loss += res.loss.item()

        # logging
        self.log("train_loss", self.train_loss / (batch_idx + 1), on_step=True, prog_bar=True, logger=True)
        self.log("train_acc_top1", acc_top1, on_step=True, prog_bar=True, logger=True)
        self.log("train_acc_top5", acc_topk, on_step=True, prog_bar=True, logger=True)

        return res.loss

    def training_epoch_end(self, training_step_outputs):
        self.train_samples = 0
        self.train_TP_top1 = 0
        self.train_TP_topk = 0
        self.train_loss = 0

    def validation_step(self, batch, batch_idx):
        # forward
        input_ids, attention_mask, labels = batch
        res = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # calculating metrics
        acc_top1, acc_topk = self._calc_acc(res.logits, labels, False)

        return {"loss": res.loss.item(), "acc_top1": acc_top1, "acc_topk": acc_topk}

    def validation_epoch_end(self, validation_step_outputs):
        epoch_acc_top1 = validation_step_outputs[-1]["acc_top1"]
        epoch_acc_topk = validation_step_outputs[-1]["acc_topk"]
        val_epoch_loss = sum(map(lambda x: x["loss"], validation_step_outputs)) / len(validation_step_outputs)

        self.log("val_loss", val_epoch_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc_top1", epoch_acc_top1, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc_top5", epoch_acc_topk, on_epoch=True, prog_bar=True, logger=True)

        self.val_samples = 0
        self.val_TP_top1 = 0
        self.val_TP_topk = 0

        return {"val_loss": val_epoch_loss, "val_acc_top1": epoch_acc_top1, "val_acc_top5": epoch_acc_topk}

    def _calc_acc(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        is_train: bool,
        top_k: int = 5,
        ignore_index: int = -100,
        shift: bool = True,
    ) -> Tuple[float, float]:
        # calc batch TP and n_samples
        assert logits.ndimension() == labels.ndimension() + 1
        assert logits.size()[:-1] == labels.size()
        assert logits.size(-1) >= top_k

        with torch.no_grad():
            if shift:
                logits = logits[..., :-1, :]
                labels = labels[..., 1:]

            labels = labels.flatten()
            logits = logits.flatten(end_dim=-2)

            relevant_labels_mask = labels != ignore_index
            labels = labels[relevant_labels_mask]
            logits = logits[relevant_labels_mask, :]

            _, top_k_preds = torch.topk(logits, top_k)

            TP_mask = top_k_preds == labels.unsqueeze(-1).expand_as(top_k_preds)

            TP_top1 = TP_mask[:, 0].sum()
            TP_topk = TP_mask.sum()

            n_samples = TP_mask.size(0)

            # accumulate metrics
            if is_train:
                TP_top1 += self.train_TP_top1
                TP_topk += self.train_TP_topk
                n_samples += self.train_samples

                self.train_TP_top1 = TP_top1
                self.train_TP_topk = TP_topk
                self.train_samples = n_samples
            else:
                TP_top1 += self.val_TP_top1
                TP_topk += self.val_TP_topk
                n_samples += self.val_samples

                self.val_TP_top1 = TP_top1
                self.val_TP_topk = TP_topk
                self.val_samples = n_samples

            acc_top1 = TP_top1 / n_samples
            acc_topk = TP_topk / n_samples
            return acc_top1, acc_topk

    def configure_optimizers(self):
        # freezing parameters
        if self.freeze_model:
            for p in self.model.transformer.parameters():
                p.requires_grad = False

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.w_decay, eps=self.eps
        )
        if self.scheduler == "cosine":
            if isinstance(self.warmup_steps, float):
                warmup_steps = self.num_training_steps * self.warmup_steps
            else:
                warmup_steps = self.warmup_steps

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer


class DataModule(pl.LightningDataModule):
    train_ds: TensorDataset
    val_ds: TensorDataset
    tokens: Dict[str, List[int]]

    def __init__(
        self,
        tokenizer_path: str,
        data_path: str,
        train_batch_size: int,
        val_batch_size: int,
        train_size: float,
        seq_len: int,
        seed: int = 42,
    ):
        super().__init__()

        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = "<pad>"

        self.data_path = data_path
        self.seq_len = seq_len

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.train_size = train_size
        self.train_steps = None

        self.seed = seed

    def prepare_data(self) -> None:
        samples = self._prep_samples(self.data_path)
        self.tokens = self.tokenizer(samples, padding=True, truncation=True, max_length=self.seq_len)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = [self.tokens["input_ids"], self.tokens["attention_mask"], self.tokens["input_ids"]]
        dataset = TensorDataset(*[torch.Tensor(x).type(torch.LongTensor) for x in dataset])

        self.train_ds, self.val_ds = train_test_split(dataset, train_size=self.train_size, random_state=self.seed)
        self.train_steps = len(self.train_ds) // self.train_batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False)

    def _prep_samples(self, data_path: str) -> List[str]:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.readlines()

        return text


@hydra.main(config_name="train-config.yaml")
def train(config: DictConfig) -> None:
    pl.seed_everything(config.seed)

    OmegaConf.set_struct(config, False)
    config.hydra_base_dir = os.getcwd()
    original_wd = hydra.utils.get_original_cwd()
    os.chdir(original_wd)

    data = DataModule(
        tokenizer_path=config.tokenizer,
        data_path=os.path.join(ROOT_PATH, config.dataset.path),
        train_batch_size=config.dataset.train_batch_size,
        val_batch_size=config.dataset.val_batch_size,
        train_size=config.dataset.train_size,
        seq_len=config.dataset.seq_len,
        seed=config.seed,
    )

    model = GPT2Sber(
        config.model,
        lr=config.training.opt.lr,
        w_decay=config.training.opt.w_decay,
        eps=config.training.opt.eps,
        warmup_steps=config.training.opt.warmup_steps,
        freeze_model=config.training.opt.freeze,
        scheduler=config.training.opt.scheduler
    )

    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.3f}",
        dirpath=config.hydra_base_dir,
        save_top_k=config.training.logging.save_top_k,
        save_last=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    stopping_callback = EarlyStopping(monitor="val_acc_top5", min_delta=1e-4, patience=3, verbose=True, mode="min")

    logger = WandbLogger(project="hse_dl_project", log_model=True)
    # TODO: This kind of logging doesn't work
    logger.log_hyperparams(dict(config))

    trainer = pl.Trainer(
        limit_train_batches=config.dataset.limit_batches,
        limit_val_batches=config.dataset.limit_batches,
        val_check_interval=config.training.logging.val_check_interval,
        max_epochs=config.training.opt.max_epochs,
        accumulate_grad_batches=config.training.opt.grad_accumulation_steps,
        gpus=config.training.n_gpus,
        gradient_clip_val=config.training.opt.grad_clip,
        auto_select_gpus=(True if config.training.n_gpus > 0 else False),
        # accelerator=("ddp" if config.training.n_gpus > 0 else None),
        log_every_n_steps=config.training.logging.log_steps,
        logger=logger,
        callbacks=[lr_logger, checkpoint_callback, stopping_callback],
        stochastic_weight_avg=config.training.opt.swa
    )
    print("Start training...")
    trainer.fit(model, datamodule=data)
    print("Finished training.")


if __name__ == "__main__":
    train()
