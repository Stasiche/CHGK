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
from src.training.pl.train import GPT2Sber, DataModule
from definitions import ROOT_PATH, SpecialTokens


@hydra.main(config_name="config.yaml")
def validate(config: DictConfig) -> None:
    model = GPT2Sber(
        os.path.join(ROOT_PATH, config.model),
        lr=config.training.opt.lr,
        w_decay=config.training.opt.w_decay,
        eps=config.training.opt.eps,
        warmup_steps=config.training.opt.warmup_steps,
        freeze_model=config.training.opt.freeze,
        scheduler=config.training.opt.scheduler
    )

    data = DataModule(
        tokenizer_path=config.tokenizer,
        data_path=os.path.join(ROOT_PATH, config.dataset.path),
        train_batch_size=config.dataset.train_batch_size,
        val_batch_size=config.dataset.val_batch_size,
        train_size=config.dataset.train_size,
        seq_len=config.dataset.seq_len,
        seed=config.seed,
    )

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
        stochastic_weight_avg=config.training.opt.swa
    )
    print("Started validation...")
    trainer.validate(model, datamodule=data, verbose=True)
    print("Finished validation.")


if __name__ == '__main__':
    validate()
