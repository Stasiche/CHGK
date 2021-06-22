import os
import hydra
import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional, Tuple, Iterable, List, Union, OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf

from src.preprocessing.get_data_generator import get_data_generator
from src.training.models.GPT2SberSmall import GPT2SberSmall
from src.training.utils.chgk_datasets import GPT2SmallDataset
from definitions import ROOT_PATH, SpecialTokens


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
        self.tokenizer.pad_token = SpecialTokens.PAD

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
        ds = self._create_dataset()
        self._split_dataset(ds)

    def _create_dataset(self) -> TensorDataset:
        ...

    def _split_dataset(self, dataset: TensorDataset):
        self.train_ds, self.val_ds = train_test_split(dataset, train_size=self.train_size, random_state=self.seed)
        self.train_steps = len(self.train_ds) // self.train_batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False)

    def _prep_samples(self, data_path: str) -> List[str]:
        return [self._gather_the_line(el) for el in get_data_generator(data_path)]

    @staticmethod
    def _gather_the_line(line: OrderedDict) -> str:
        return " ".join(
            [
                SpecialTokens.BOS.value,
                line["answer"].strip(),
                SpecialTokens.ANS.value,
                line["question"].strip(),
                SpecialTokens.EOS.value,
            ]
        )


class CSVDataModule(DataModule):
    def __init__(self, ans_seq_len: int = 20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ans_seq_len = ans_seq_len

        self.bos_id, self.ans_delim_id, self.eos_id = \
            self.tokenizer.convert_tokens_to_ids([SpecialTokens.BOS.value, SpecialTokens.ANS.value, SpecialTokens.EOS.value])

    def _create_dataset(self) -> TensorDataset:
        csv = pd.read_csv(self.data_path)

        ans_encoded = self.tokenizer(csv["answer"].values.tolist(), padding=True, truncation=True, max_length=self.ans_seq_len,
                                     return_tensors="pt")
        q_encoded = self.tokenizer(csv["question"].values.tolist(), padding=True, truncation=True, max_length=self.seq_len,
                                   return_tensors="pt")

        n_samples = len(csv)
        bos_ids = torch.ones(n_samples).reshape(n_samples, 1).type(torch.LongTensor)
        ans_del_ids = torch.ones(n_samples).reshape(n_samples, 1).type(torch.LongTensor)
        eos_ids = torch.ones(n_samples).reshape(n_samples, 1).type(torch.LongTensor)

        input_ids = torch.cat(
            [bos_ids * self.bos_id, ans_encoded["input_ids"], ans_del_ids * self.ans_delim_id, q_encoded["input_ids"], eos_ids * self.eos_id],
            dim=-1)

        att_mask = torch.cat(
            [bos_ids, ans_encoded["attention_mask"], ans_del_ids, q_encoded["attention_mask"], eos_ids],
            dim=-1)

        dataset = TensorDataset(input_ids, att_mask, input_ids)

        return dataset


