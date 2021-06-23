import dfa as dfa
from parse.intent import Intent, Command


class GoodbyeState(dfa.BaseState):
    def __init__(self):
        super().__init__()

    def move(self, intent: Intent) -> dfa.MoveResponse:
        if "greetings" in intent.parameters:
            intent.command = Command.GREETINGS
            return dfa.MoveResponse(dfa.GreetingsState(), "Привет! Рад снова тебя видеть!")
        else:
            return dfa.MoveResponse(self, "Нет привета -- нет ответа!")

