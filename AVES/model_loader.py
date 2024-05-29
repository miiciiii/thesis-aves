import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sense2vec import Sense2Vec
import pandas as pd

# File paths
DATA_PATH = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_datasets\generated_qa.csv"
PASSAGE_PATH = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_datasets\news.csv"
T5QG_MODEL_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\qg_model"
T5QG_TOKENIZER_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\qg_tokenizer"
T5AG_MODEL_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\t5_model"
T5AG_TOKENIZER_DIR = r"D:\02 Personal Files\Thesis Related\Main Thesis Project\IRC-AVES\IRC_AVES_project\AVES_models\t5_tokenizer"
S2V_MODEL_PATH = 's2v_old'

# Lazy load models
data = pd.read_csv(DATA_PATH)
passage = pd.read_csv(PASSAGE_PATH)
t5ag_model = None
t5ag_tokenizer = None
t5qg_model = None
t5qg_tokenizer = None
summary_model = None
summary_tokenizer = None
s2v = None
sentence_transformer_model = None

def load_models():
    global t5ag_model, t5ag_tokenizer, t5qg_model, t5qg_tokenizer, summary_model, summary_tokenizer, s2v, sentence_transformer_model
    if t5ag_model is None:
        t5ag_model = T5ForConditionalGeneration.from_pretrained(T5AG_MODEL_DIR)
    if t5ag_tokenizer is None:
        t5ag_tokenizer = T5Tokenizer.from_pretrained(T5AG_TOKENIZER_DIR)
    if t5qg_model is None:
        t5qg_model = T5ForConditionalGeneration.from_pretrained(T5QG_MODEL_DIR)
    if t5qg_tokenizer is None:
        t5qg_tokenizer = T5Tokenizer.from_pretrained(T5QG_TOKENIZER_DIR)
    if summary_model is None:
        summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    if summary_tokenizer is None:
        summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    if s2v is None:
        s2v = Sense2Vec().from_disk(S2V_MODEL_PATH)
    if sentence_transformer_model is None:
        sentence_transformer_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")

