from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset
from typing import List


class GPT2SmallDataset(Dataset):
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

    def __init__(self, txt_path: str):
        tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        tokenizer.pad_token = "<pad>"

        samples = self._prep_samples(txt_path)
        self.tokens = tokenizer(samples, padding=True, truncation=True, max_length=206)

        print('Done tokenizing')

    def __len__(self):
        return len(self.tokens["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens["input_ids"][idx],
            "attention_mask": self.tokens["attention_mask"][idx],
            "labels": self.tokens["input_ids"][idx],
        }

    def _prep_samples(self, data_path: str) -> List[str]:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.readlines()
            
        return text


