import dfa as dfa
from typing import Optional
from parse.intent import Intent, Command


class AskSearchThemeState(dfa.BaseState):
    def __init__(self):
        super().__init__()

    @property
    def introduce_message(self) -> Optional[str]:
        return "Какая тема тебя интересует?"
    
    def move(self, intent: Intent) -> dfa.MoveResponse:
        """
        не надо парсить сообщение -> сразу запишем и перейдем
        """
        if "bye" in intent.parameters:
            intent.command = Command.GOODBYE
            return dfa.MoveResponse(dfa.GoodbyeState(), "Пока-пока!")
        
        intent.parameters["theme"] = intent.message
        intent.command = Command.SEARCH
        return dfa.MoveResponse(dfa.GetQuestionFromDBState(), None)
    