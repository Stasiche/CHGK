import dfa as dfa
from parse.intent import Intent, Command
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import pickle
import numpy 


tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
model.to("cpu").eval()


with open("/home/breengles/Dropbox/hse/CHKG_proj/bot/dfa/search/kmeans.pkl", "rb") as km:
    kmeans = pickle.load(km)
    

db = pd.read_csv("/home/breengles/Dropbox/hse/CHKG_proj/bot/dfa/search/cleaned_clusters.csv")


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
            message = get_question_from_db(intent.parameters['theme'])
            return dfa.MoveResponse(next_state, message)
        return dfa.MoveResponse(dfa.AskSearchThemeState(), None)


def get_intent_embedding(theme):
    token = tokenizer(theme, return_tensors="pt", padding=True, truncation=True, max_length=100)
    with torch.no_grad():
        model_dict = model(**token)
    return model_dict["last_hidden_state"][:, 0, :].cpu().numpy()


def get_question_from_db(theme):
    theme_emb = get_intent_embedding(theme)
    theme_clstr = kmeans.predict(theme_emb)
    possible_question = db[db["cluster_id"] == theme_clstr[0]]["question"].sample().values[0]
    return possible_question
