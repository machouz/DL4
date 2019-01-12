from __future__ import unicode_literals
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F
from utils import *

EPOCHS = 5
HIDDEN_RNN = [50, 50]
CHAR_LSTM = 50
EMBEDDING = 50
BATCH_SIZE = 100
LR = 0.01
LR_DECAY = 0.5
EMBEDDING_PROJECTION = 256
GLOVE_DIM = 300
FAST_TEXT_DIM = 300


def get_embedding(pretrained_embedding, embedding_size, vocab2id):
    embedding = nn.Embedding(num_embeddings=len(vocab2id), embedding_dim=embedding_size)
    data = embedding.weight.data
    for word, id in vocab2id.items():
        if word in pretrained_embedding:
            data[id] = pretrained_embedding[word].mean(0)
        elif word.lower() in pretrained_embedding:
            data[id] = pretrained_embedding[word.lower()].mean(0)
        else:
            data[id] = 0
    return embedding


class UnweightedDME(nn.Module):
    def __init__(self, glove_path, fast_text_path, vocab2id):
        super(UnweightedDME, self).__init__()
        self.glove = get_embedding(glove_path, GLOVE_DIM, vocab2id)
        self.fast_text = get_embedding(fast_text_path, FAST_TEXT_DIM, vocab2id)
        self.P_glove = nn.Linear(GLOVE_DIM, EMBEDDING_PROJECTION)
        self.P_fast_text = nn.Linear(FAST_TEXT_DIM, EMBEDDING_PROJECTION)

    def forward(self, word):
        emb_glove = self.glove(word)
        emb_fast_text = self.fast_text(word)
        out_glove = self.P_glove(emb_glove)
        out_fast_text = self.P_fast_text(emb_fast_text)
        stacked = torch.stack([out_glove, out_fast_text])
        output = torch.sum(stacked)
        return output


if __name__ == '__main__':

    glove_path = '../checkpoints/cache/glove.840B.300d.txt.pkl'
    pretrained_embedding = load_embedding(glove_path)

    vocab_path = '../checkpoints/cache/vocab.pkl'
    vocab2id = get_vocab2id(vocab_path)

