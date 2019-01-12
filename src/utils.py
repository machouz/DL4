import pickle


def load_embedding(path):
    with open(path, 'rb') as e:
        embedding = pickle.load(e, encoding='bytes')
    return embedding


def get_embeds_vocab(path):
    with open(path, 'rb') as v:
        vocab = pickle.load(v)
    vocab2id = {word: i for i, word in enumerate(list(vocab))}
    return vocab2id


