import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from nn_models.GPT2SberSmall import GPT2SberSmall


tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
rubert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
rubert.to("cpu").eval()


with open("db/kmeans.pkl", "rb") as km:
    kmeans = pickle.load(km)
    

db = pd.read_csv("db/question_cluster.csv")

def get_intent_embedding(theme):
    token = tokenizer(theme, return_tensors="pt", padding=True, truncation=True, max_length=100)
    with torch.no_grad():
        model_dict = rubert(**token)
    return model_dict["last_hidden_state"][:, 0, :].cpu().numpy().astype(float)

rugpt = GPT2SberSmall("/home/breengles/Dropbox/projects/CHGK/models",  # this is some hardcoding magic!
                      "sberbank-ai/rugpt3small_based_on_gpt2", "cuda")
rugpt.eval()