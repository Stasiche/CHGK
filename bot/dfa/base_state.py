from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Callable, Optional
from parse.intent import Intent, Command
from singleton import Singleton
import re

STRELOK = re.compile(r"стрел(ок|ку)")
STALKER = re.compile(r"хабару?")
ANEKDOT = re.compile(r"какая машина самая сталкерская")

@dataclass
class MoveResponse:
    new_state: "BaseState"
    message: Optional[str]


IntentHandler = Callable[[Intent], MoveResponse]


class BaseState(metaclass=Singleton):
    def __init__(self):
        self._logger = getLogger(__file__)
        self._command_handler: Dict[Command, IntentHandler] = {}

    @property
    def introduce_message(self) -> Optional[str]:
        return None

    @property
    def is_technical_state(self) -> bool:
        return False

    def move(self, intent: Intent) -> MoveResponse:
        with open("log/log.txt", "a+") as log:
            log.write(f"{intent.message} | {intent.command} | {intent.parameters} \n")
            
        if intent.command in self._command_handler:
            return self._command_handler[intent.command](intent)
        
        if STRELOK.search(intent.message.lower()):
            return MoveResponse(self, "https://www.youtube.com/watch?v=_VxQrbNPxO4")
        
        if STALKER.search(intent.message.lower()):
            return MoveResponse(self, "Короче, Меченый, я тебя спас и в благородство играть не буду: выполнишь для меня пару заданий — и мы в расчете. Заодно посмотрим, как быстро у тебя башка после амнезии прояснится. А по твоей теме постараюсь разузнать. Хрен его знает, на кой ляд тебе этот Стрелок сдался, но я в чужие дела не лезу, хочешь убить, значит есть за что...")
        
        if ANEKDOT.search(intent.message.lower()):
            return MoveResponse(self, "Запорожец! Потому что надежная, как танк, и багажник спереди: за хабаром удобно присматривать!")
        
        if intent.message.isnumeric():
            return MoveResponse(self, "С роботами не разговариваю!")
        
        return self.handle_unknown_command()

    def handle_unknown_command(self) -> MoveResponse:
        return MoveResponse(self, "Прости, я тебя не понимаю")



