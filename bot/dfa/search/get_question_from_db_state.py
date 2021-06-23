import dfa as dfa
from parse.intent import Intent, Command
import torch
import numpy
from db.utils import kmeans, db, get_intent_embedding
from parse.keywords import autostop


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
            if autostop.search(intent.message.lower()):
                message = "Вопрос ты уже задал, а ответ и так знаешь: 42"
            else:
                message = get_question_from_db(intent.parameters['theme'])
            return dfa.MoveResponse(next_state, message)
        return dfa.MoveResponse(dfa.AskSearchThemeState(), None)


def get_question_from_db(theme):
    theme_emb = get_intent_embedding(theme)
    theme_clstr = kmeans.predict(theme_emb)
    possible_question = db[db["cluster_id"] == theme_clstr[0]]["question"].sample().values[0]
    return possible_question
