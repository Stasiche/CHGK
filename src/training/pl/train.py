import os
import hydra
import pytorch_lightning as pl
import torch
import numpy as np

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Dict, Any, Optional, Tuple, Iterable, List
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from src.training.models.GPT2SberSmall import GPT2SberSmall
from src.training.utils.chgk_datasets import GPT2SmallDataset
from definitions import ROOT_PATH


class GPT2Sber(pl.LightningModule):
    def __init__(self, model_path: str, lr: float, w_decay: float):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.train_TP = 0
        self.val_TP = 0

        self.train_samples = 0
        self.val_samples = 0

        self.train_loss = 0

        self.lr = lr
        self.w_decay = w_decay

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
        train_acc = self._calc_acc(res.logits, labels, True)
        self.train_loss += res.loss.item()

        # logging
        self.log("train_loss", self.train_loss / (batch_idx + 1), on_step=True, prog_bar=True, logger=True)
        self.log("train_acc", train_acc, on_step=True, prog_bar=True, logger=True)

        return res.loss

    def training_epoch_end(self, training_step_outputs):
        self.train_samples = 0
        self.train_TP = 0

    def validation_step(self, batch, batch_idx):
        # forward
        input_ids, attention_mask, labels = batch
        res = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # calculating metrics
        val_acc = self._calc_acc(res.logits, labels, False)

        return {"loss": res.loss.item(), "acc": val_acc}

    def validation_epoch_end(self, validation_step_outputs):
        val_epoch_acc = validation_step_outputs[-1]["acc"]
        val_epoch_loss = sum(map(lambda x: x["loss"], validation_step_outputs)) / len(validation_step_outputs)

        self.log("val_loss", val_epoch_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", val_epoch_acc, on_epoch=True, prog_bar=True, logger=True)

        self.val_samples = 0
        self.val_TP = 0

        return {"val_loss": val_epoch_loss, "val_acc": val_epoch_acc}

    def _calc_acc(self, logits, labels, is_train) -> float:
        # calc batch TP and n_samples
        with torch.no_grad():
            preds = logits.argmax(dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()

        relevant_labels_mask = labels != -100
        TP_mask = preds[relevant_labels_mask] == labels[relevant_labels_mask]

        TP = np.count_nonzero(TP_mask)
        n_samples = np.prod(TP_mask.shape)

        # accumulate metrics
        if is_train:
            TP += self.train_TP
            n_samples += self.train_samples

            self.train_TP += TP
            self.train_samples += n_samples
        else:
            TP += self.val_TP
            n_samples += self.val_samples

            self.val_TP = TP
            self.val_samples = n_samples

        acc = TP / n_samples
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        return optimizer


class DataModule(pl.LightningDataModule):
    train_ds: TensorDataset
    val_ds: TensorDataset
    tokens: Dict[str, List[int]]

    def __init__(self, tokenizer_path: str, data_path: str, train_batch_size: int, val_batch_size: int, train_size: float):
        super().__init__()

        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = "<pad>"

        self.data_path = data_path

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.train_size = train_size

    def prepare_data(self) -> None:
        samples = self._prep_samples(self.data_path)
        self.tokens = self.tokenizer(samples, padding=True, truncation=True, max_length=206)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = [self.tokens["input_ids"], self.tokens["attention_mask"], self.tokens["input_ids"]]
        dataset = TensorDataset(*[torch.Tensor(x).type(torch.LongTensor) for x in dataset])

        self.train_ds, self.val_ds = train_test_split(dataset, train_size=self.train_size)

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

    model = GPT2Sber(config.model, lr=config.training.params.lr, w_decay=config.training.params.w_decay)

    data = DataModule(
        tokenizer_path=config.tokenizer,
        data_path=os.path.join(ROOT_PATH, config.dataset.path),
        train_batch_size=config.dataset.train_batch_size,
        val_batch_size=config.dataset.val_batch_size,
        train_size=config.dataset.train_size
    )

    logger = WandbLogger(project="hse_dl_project", log_model=False)

    trainer = pl.Trainer(
        val_check_interval=config.training.logging.val_check_interval,
        max_epochs=config.training.params.max_epochs,
        gpus=config.training.n_gpus,
        logger=logger,
    )
    print("Start training...")
    trainer.fit(model, datamodule=data)
    print("Finished training.")


if __name__ == "__main__":
    train()
