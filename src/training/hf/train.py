from transformers import Trainer, TrainingArguments
from datasets import load_metric
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from src.training.models.GPT2SberSmall import GPT2SberSmall
from src.training.utils.chgk_datasets import GPT2SmallDataset
from definitions import ROOT_PATH

import hydra
import os

import numpy as np


def compute_metrics(preds):
    target = preds.label_ids.flatten()
    preds = np.argmax(preds.predictions[0], axis=-1).flatten()
    metric = load_metric("accuracy")

    metric.add_batch(predictions=preds, references=target)
    res = metric.compute()
    return res


@hydra.main(config_name="train-config.yaml")
def train(config: DictConfig) -> None:
    model = GPT2SberSmall(config.model)
    tokenizer = model.tokenizer
    dataset = GPT2SmallDataset(os.path.join(ROOT_PATH, config.dataset.path))
    dataset_train, dataset_eval = train_test_split(dataset, train_size=0.90)

    # define training params
    train_batch_size = config.dataset.train_batch_size
    n_steps = len(dataset_train) / train_batch_size
    print(f'Steps to be made: {n_steps}')

    val_steps = int(n_steps / 5)

    training_args = TrainingArguments(
        learning_rate=config.training.params.lr,
        weight_decay=config.training.params.w_decay,
        num_train_epochs=config.training.params.n_epochs,
        per_device_train_batch_size=config.dataset.train_batch_size,
        per_device_eval_batch_size=config.dataset.eval_batch_size,
        save_steps=config.training.logging.eval_steps,
        eval_steps=config.training.logging.eval_steps,
        logging_steps=config.training.logging.log_steps,
        save_total_limit=config.training.logging.save_models,
        overwrite_output_dir=True,
        output_dir=config.training.output_dir,
        evaluation_strategy='steps',
        logging_strategy='steps',
        warmup_steps=300,
        seed=config.seed,
        report_to="none",
        no_cuda=config.training.n_gpus == 0,
        # load_best_model_at_end=True,
        # metric_for_best_model='accuracy',
        # greater_is_better=True,
    )
    print('Created args')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        # compute_metrics=compute_metrics
    )
    print('Created trainer')

    print('Start training')
    trainer.train()
    print('Finished training')


if __name__ == '__main__':
    train()
