from dfa.base_state import MoveResponse
import dfa as dfa
from parse.intent import Intent, Command


class StartState(dfa.BaseState):
    def __init__(self):
        super().__init__()
        self._command_handler[Command.QUESTION] = self.handle_question_command
        self._command_handler[Command.GAME] = self.handle_game_command
        self._command_handler[Command.SEARCH] = self.handle_search_command
        self._command_handler[Command.GOODBYE] = self.handle_goodbye_command
        self._command_handler[Command.GREETINGS] = self.handle_greetings_command

    def handle_question_command(self, intent: Intent) -> dfa.MoveResponse:
        return dfa.MoveResponse(dfa.GetQuestionState(), None)

    def handle_game_command(self, intent: Intent) -> dfa.MoveResponse:
        return dfa.MoveResponse(dfa.GetGameState(), None)
    
    def handle_search_command(self, intent: Intent) -> dfa.MoveResponse:
        return dfa.MoveResponse(dfa.GetQuestionFromDBState(), None)

    def handle_goodbye_command(self, intent: Intent) -> dfa.MoveResponse:
        if "bye" in intent.parameters:
            intent.command = Command.GOODBYE
            return dfa.MoveResponse(dfa.GoodbyeState(), "Пока-пока!")
    
    def handle_greetings_command(self, intent: Intent) -> dfa.MoveResponse:
        if "greetings" in intent.parameters:
            intent.command = Command.GREETINGS
            return dfa.MoveResponse(dfa.GreetingsState(), "Привет!")
    
    