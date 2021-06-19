import os
import time

import torch

from argparse import ArgumentParser
from src.training.models.GPT2SberSmall import GPT2SberSmall
from definitions import SBER_MODEL_SMALL, SpecialTokens
from transformers import logging
from typing import Tuple


def generate(
    model_dir: str,
    tokenizer_path: str,
    context: str,
    max_len: int,
    beam_size: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[str, float]:
    model = GPT2SberSmall(model_dir, tokenizer_path, device)
    model.eval()

    start = time.time()
    generated_text = model.generate(context, max_len, beam_size)
    elapsed_time = time.time() - start

    return generated_text, elapsed_time


if __name__ == "__main__":
    logging.set_verbosity(0)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    args = ArgumentParser()
    args.add_argument("--model_dir", required=False, type=str, default=SBER_MODEL_SMALL, help="Directory with hf model")
    args.add_argument(
        "--tokenizer_path", required=False, type=str, default=SBER_MODEL_SMALL, help="Directory with tokenizer"
    )
    args.add_argument("--context", type=str, required=True, help="Context for generation")
    args.add_argument("--max_len", type=int, required=False, default=50, help="Max length of the output in tokens")
    args.add_argument("--beam_size", type=int, required=False, default=5, help="Beam width")
    args.add_argument("--gpu", action="store_true", help="Whether to use GPU or not")
    args = args.parse_args()

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    generated_text, elapsed_time = generate(
        args.model_dir, args.tokenizer_path, args.context, args.max_len, args.beam_size, device
    )
    print(generated_text)
    print(f"Elapsed time: {elapsed_time: .3f} seconds")
