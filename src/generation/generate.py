import os
import time

import torch

from argparse import ArgumentParser
from src.training.models.GPT2SberAbstract import GPT2SberContext, GPT2SberSimple
from definitions import SBER_MODEL_SMALL, SpecialTokens
from transformers import logging
from typing import Tuple


def generate(
    model_dir: str,
    tokenizer_path: str,
    context: str,
    max_len: int,
    beam_size: int,
    top_k: int,
    top_p: float,
    is_answer: bool,
    sampling: bool,
    device: torch.device = torch.device("cpu"),
) -> Tuple[str, float]:
    if not is_answer:
        model = GPT2SberSimple(model_dir, tokenizer_path, device)
    else:
        model = GPT2SberContext(model_dir, tokenizer_path, device)
    model.eval()

    start = time.time()
    generated_text = model.generate(
        context=context, max_length=max_len, beam_size=beam_size, sampling=sampling, top_k=top_k, top_p=top_p
    )
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
    args.add_argument("--max_len", type=int, required=False, default=100, help="Max length of the output in tokens")
    args.add_argument("--beam_size", type=int, required=False, default=5, help="Beam width")
    args.add_argument("--gpu", action="store_true", help="Whether to use GPU or not")
    args.add_argument("--ans", action="store_true", help="Model type -- with answers or not")
    args.add_argument("--sampling", action="store_true", help="Generation type")
    args.add_argument("--top_k", type=int, default=50)
    args.add_argument("--top_p", type=float, default=0.9)
    args = args.parse_args()

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    generated_text, elapsed_time = generate(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        context=args.context,
        max_len=args.max_len,
        beam_size=args.beam_size,
        is_answer=args.ans,
        sampling=args.sampling,
        device=device,
        top_k=args.top_k,
        top_p=args.top_p
    )
    print(generated_text)
    print(f"Elapsed time: {elapsed_time: .3f} seconds")
