import torch
from six.moves import cPickle as pickle


def load_embedding(path):
    with open(path, 'rb') as e:
        if torch.cuda.is_available():
            embedding = pickle.load(e, encoding='bytes')
        else:
            embedding = pickle.load(e, encoding='bytes')
    return embedding


def get_embeds_vocab(path):
    with open(path, 'rb') as v:
        vocab = pickle.load(v, encoding='bytes')
    vocab2id = {word: i for i, word in enumerate(list(vocab))}
    return vocab2id


def get_vocab2id(path):
    with open(path, 'rb') as v:
        vocab2id = pickle.load(v)
    return vocab2id


def get_emb_key(word, embeds):
    if word in embeds:
        return word, 'exact'
    elif word.lower() in embeds:
        return word.lower(), 'lower'
    else:
        return None, ''