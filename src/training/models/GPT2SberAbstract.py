import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from definitions import SpecialTokens


class GPT2SberAbstract(torch.nn.Module):
    def __init__(self, model_dir: str, tokenizer_path: str, device: torch.device):
        super().__init__()
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

        self.model.to(device)
        self.device = device

        self.tokenizer.pad_token = SpecialTokens.PAD.value
        self.tokenizer.bos_token = SpecialTokens.BOS.value
        self.tokenizer.eos_token = SpecialTokens.EOS.value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    @torch.no_grad()
    def generate(self, context: str, max_length: int, beam_size: int):
        input_ids = self._pre_example(context)

        greedy_output = self.model.generate(
            input_ids, max_length=max_length, beam_size=beam_size, no_repeat_ngram_size=2, early_stopping=True
        )
        generated_output = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        generated_output = self._post_example(generated_output)

        return generated_output

    def _pre_example(self, example) -> torch.LongTensor:
        ...

    def _post_example(self, generated_text) -> str:
        ...


class GPT2SberSimple(GPT2SberAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _pre_example(self, example) -> torch.LongTensor:
        example = SpecialTokens.BOS.value + " " + example
        input_ids = self.tokenizer.encode(example, return_tensors="pt").to(self.device)

        return input_ids

    def _post_example(self, generated_text) -> str:
        eos_ind = generated_text.find("</s")
        if eos_ind == -1:
            eos_ind = None

        bos_len = len(SpecialTokens.BOS.value) + 1
        generated_text = generated_text[bos_len:eos_ind]

        return generated_text


class GPT2SberContext(GPT2SberAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _pre_example(self, example) -> torch.LongTensor:
        example = " ".join([SpecialTokens.BOS.value, example, SpecialTokens.ANS.value])
        example += " "
        input_ids = self.tokenizer.encode(example, return_tensors="pt").to(self.device)

        # context_ids = self.tokenizer.encode(example)
        # bos_id, delim_id = self.tokenizer.convert_tokens_to_ids([SpecialTokens.BOS.value, SpecialTokens.ANS.value])
        #
        # input_ids = torch.tensor([bos_id] + context_ids + [delim_id]).unsqueeze(0).type(torch.LongTensor)

        return input_ids.to(self.device)

    def _post_example(self, generated_text) -> str:
        # return generated_text
        eos_ind = generated_text.find("</s")
        q_start_ind = generated_text.find(SpecialTokens.ANS.value)
        print(q_start_ind)
        if eos_ind == -1:
            eos_ind = None

        ans_token_len = len(SpecialTokens.ANS.value)
        generated_text = generated_text[q_start_ind+ans_token_len:eos_ind]

        return generated_text
