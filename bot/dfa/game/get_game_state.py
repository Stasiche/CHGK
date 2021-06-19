import dfa as dfa
from parse.intent import Intent, Command


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
            message = f"Creating game"
            return dfa.MoveResponse(next_state, message)
        return dfa.MoveResponse(next_state, None)
