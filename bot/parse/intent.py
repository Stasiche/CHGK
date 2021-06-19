from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict


class Command(Enum):
    GREETINGS = auto()
    QUESTION = auto()
    GAME = auto()
    SEARCH = auto()
    GOODBYE = auto()
    UNKNOWN = auto()
    # START = auto()


@dataclass
class Intent:
    command: Command
    message: str
    parameters: Dict[str, str] = field(default_factory=dict, init=True)
