from sentence_transformers import SentenceTransformer

MODEL_NAME = 'bert-large-nli-stsb-mean-tokens'
MODEL = SentenceTransformer(MODEL_NAME)


def embed(text):
    return MODEL.encode(text)
