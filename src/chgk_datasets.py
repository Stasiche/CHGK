from transformers import GPT2Tokenizer
from torch.utils.data import Dataset


class GPT2SmallDataset(Dataset):
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

    def __init__(self, txt_path):
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = 0
        with open(txt_path, 'r') as f:
            text = f.readlines()
        self.tokens = tokenizer(text, padding=True, truncation=True, max_length=106)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
                'input_ids': self.tokens['input_ids'][idx],
                'attention_mask': self.tokens['attention_mask'][idx],
               }