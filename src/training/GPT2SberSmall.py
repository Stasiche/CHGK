from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch


class GPT2SberSmall(torch.nn.Module):
    def __init__(self, hf_model: str):
        super().__init__()
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(hf_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(hf_model)

        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, context: str):
        input_ids = self.tokenizer.encode(context, return_tensors='pt')
        greedy_output = self.model.generate(input_ids,
                                            max_length=80,
                                            beam_size=5,
                                            no_repeat_ngram_size=2,
                                            early_stopping=True)
        generated_output = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)

        return generated_output
