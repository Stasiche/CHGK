import dfa as dfa
from parse.intent import Intent, Command


class GreetingsState(dfa.StartState):
    def __init__(self):
        super().__init__()

    @property
    def is_technical_state(self) -> bool:
        return True

    def move(self, intent: Intent) -> dfa.MoveResponse:
        if "greetings" in intent.parameters:
            if "game" in intent.parameters:
                intent.command = Command.GAME
                return dfa.MoveResponse(dfa.GetGameState(), None)
            if "generation" in intent.parameters:
                intent.command = Command.QUESTION
                return dfa.MoveResponse(dfa.GetQuestionState(), None)
            if "search" in intent.parameters:
                intent.command = Command.SEARCH
                return dfa.MoveResponse(dfa.GetQuestionFromDBState(), None)
            # intent.command = Command.START
            return dfa.MoveResponse(dfa.StartState(), None)
        return self.handle_unknown_command()


