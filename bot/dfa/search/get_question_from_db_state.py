import dfa as dfa
from parse.intent import Intent, Command


class GetQuestionFromDBState(dfa.BaseState):
    def __init__(self):
        super().__init__()
        self._command_handler[Command.SEARCH] = self.handle_question_command

    @property
    def is_technical_state(self) -> bool:
        return True

    def __disable_move(self, intent: Intent) -> dfa.MoveResponse:
        return dfa.MoveResponse(dfa.StartState(), self.__disable_message)

    def handle_question_command(self, intent: Intent) -> dfa.MoveResponse:
        if "theme" in intent.parameters:
            next_state = dfa.StartState()
            message = f"Searching DB for question `{intent.parameters['theme']}`"
            return dfa.MoveResponse(next_state, message)
        return dfa.MoveResponse(dfa.AskSearchThemeState(), None)
