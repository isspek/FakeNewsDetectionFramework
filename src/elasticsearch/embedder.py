from sentence_transformers import SentenceTransformer

MODEL_NAME = 'allenai/scibert_scivocab_uncased'
MODEL = SentenceTransformer(MODEL_NAME)


def embed(sentence):
    return MODEL.encode(sentence).tolist()  # make it compatible with Elasticsearch
