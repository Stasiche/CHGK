from transformers.models.tapas.tokenization_tapas import Question
import dfa as dfa
from parse.intent import Intent, Command
from db.utils import db



class GetGameState(dfa.BaseState):
    def __init__(self):
        super().__init__()
        self._command_handler[Command.GAME] = self.handle_game_command

    @property
    def is_technical_state(self) -> bool:
        return True

    def __disable_move(self, intent: Intent) -> dfa.MoveResponse:
        return dfa.MoveResponse(dfa.StartState(), self.__disable_message)

    def handle_game_command(self, intent: Intent) -> dfa.MoveResponse:
        next_state = dfa.StartState()
        if "game" in intent.parameters:
            message = create_game()
            return dfa.MoveResponse(next_state, message)
        return dfa.MoveResponse(next_state, None)


def create_game():
    out = ""
    questions = db.sample(12).reset_index()
    for idx, question in questions.iterrows():
        if idx == 0:
            out = out + f"{idx + 1}) {question['question']}\nОтвет: {question['answer']}"
        else:
            out = out + f"\n\n{idx + 1}) {question['question']}\nОтвет: {question['answer']}"
    return out
