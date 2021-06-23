import dfa as dfa
from parse.intent import Intent, Command
from db.utils import db, kmeans, get_intent_embedding
import torch
from typing import Tuple
from db.utils import rugpt
from parse.keywords import autostop


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
            if autostop.search(intent.message.lower()):
                message = "Вопрос ты уже задал, а ответ и так знаешь: 42"
            else:
                message = get_generated_question(intent.parameters["theme"])
            return dfa.MoveResponse(next_state, message)
        return dfa.MoveResponse(dfa.AskGenerationThemeState(), None)


def get_generated_question(theme, length=5) -> str:
    emb = get_intent_embedding(theme)
    clst = kmeans.predict(emb)
    context = db[db["cluster_id"] == clst[0]]["question"].sample().values[0].split()[:length]
    context = ' '.join(context)
    
    return generate(context=context, 
                    model_dir="/home/breengles/Dropbox/hse/CHKG_proj/bot/nn_models",
                    tokenizer_path="sberbank-ai/rugpt3small_based_on_gpt2",
                    max_len=150, beam_size=5, device="cuda")


def generate(model_dir: str, tokenizer_path: str, context: str, max_len: int,
             beam_size: int, device: torch.device = torch.device("cpu")) -> Tuple[str, float]:

    with torch.no_grad():
        context = "<s>" + " " + context

        generated_text = rugpt.generate(context, max_len, beam_size)

        eos_ind = generated_text.find("</s")
        if eos_ind == -1:
            eos_ind = None

        bos_len = len("<s>") + 1  # deploy version
        generated_text = generated_text[bos_len:eos_ind]
        
        # bos_len = len("<s>") + len(context) - 3
        # generated_text = context + " | " + generated_text[bos_len:eos_ind]

    return generated_text

