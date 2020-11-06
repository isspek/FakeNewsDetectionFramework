from sentence_transformers import SentenceTransformer

MODEL_NAME = 'bert-base-nli-mean-tokens'
MODEL = SentenceTransformer(MODEL_NAME)


def embed(sentence):
    return MODEL.encode(sentence).tolist()  # make it compatible with Elasticsearch
