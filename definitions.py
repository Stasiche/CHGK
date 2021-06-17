import os
from enum import unique, Enum

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
SBER_MODEL_SMALL = "sberbank-ai/rugpt3small_based_on_gpt2"


@unique
class SpecialTokens(Enum):
    PAD = "<pad>"
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"
