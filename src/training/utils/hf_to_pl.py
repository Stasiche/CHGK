import argparse
import os

from src.training.pl.train import GPT2Sber


def convert(pl_checkpoint_path: str, hf_output_path: str) -> None:
    pl_model = GPT2Sber.load_from_checkpoint(checkpoint_path=pl_checkpoint_path)
    pl_model.model.save_pretrained(hf_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pl_checkpoint_path", required=True, type=str, help="PL checkpoint path"
    )
    parser.add_argument(
        "-o", "--hf_output_path", required=True, type=str, help="Directory for storing HF checkpoint",
    )
    args = parser.parse_args()

    assert not os.path.exists(args.hf_output_path), "Target directory already exists"
    convert(args.pl_checkpoint_path, args.hf_output_path)
