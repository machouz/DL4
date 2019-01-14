from six.moves import cPickle as pickle
import torch
from torchtext import data

def load_embedding(path):
    with open(path, 'rb') as e:
        if torch.cuda.is_available():
            embedding = pickle.load(e, encoding='bytes')
        else:
            embedding = pickle.load(e)
    return embedding


def get_embeds_vocab(path):
    with open(path, 'rb') as v:
        vocab = pickle.load(v)
    vocab2id = {word: i for i, word in enumerate(list(vocab))}
    return vocab2id


def get_vocab2id(path):
    with open(path, 'rb') as v:
        vocab2id = pickle.load(v)
    return vocab2id


