import os

from argparse import ArgumentParser
from src.training.models.GPT2SberSmall import GPT2SberSmall
from definitions import SBER_MODEL_SMALL
from transformers import logging


def generate(model_dir: str, tokenizer_path: str, context: str, max_len: int, beam_size: int) -> str:
    model = GPT2SberSmall(model_dir, tokenizer_path)
    generated_text = model.generate(context, max_len, beam_size)

    return generated_text


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
    args = args.parse_args()

    generated_text = generate(args.model_dir, args.tokenizer_path, args.context, args.max_len, args.beam_size)
    print(generated_text)
