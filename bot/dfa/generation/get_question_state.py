import dfa as dfa
from parse.intent import Intent, Command


class GetQuestionState(dfa.BaseState):
    def __init__(self):
        super().__init__()
        self._command_handler[Command.QUESTION] = self.handle_question_command

    @property
    def is_technical_state(self) -> bool:
        return True

    def handle_question_command(self, intent: Intent) -> dfa.MoveResponse:
        if "theme" in intent.parameters:
            next_state = dfa.StartState()
            message = f"Generating question `{intent.parameters['theme']}`"
            return dfa.MoveResponse(next_state, message)
        return dfa.MoveResponse(dfa.AskGenerationThemeState(), None)
