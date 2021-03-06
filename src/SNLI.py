from __future__ import unicode_literals
import torch
import torch.nn as nn
from classifier import *
from dynamic_meta_embeddings import *
from sentence_encoder import *
import numpy as np

VOCAB_PATH = 'cache/vocab.pkl'
GLOVE_PATH = 'cache/matched_glove_.pkl'
FAST_TEXT_PATH = 'cache/matched_crawl_.pkl'
LEVY_DEP_PATH = 'cache/matched_levy_.pkl'
GLOVE_DIM = 300
FAST_TEXT_DIM = 300
EMBEDDING_PROJECTION = 256


LSTM_DIM = 512
MLP_HIDDEN_LAYER = 1024
DROPOUT = 0.2
TAGSET_SIZE = 3

def count_param_num(nn_module):
    return np.sum([np.prod(param.size()) for param in nn_module.parameters() if param.requires_grad])

class SNLI(nn.Module):

    def __init__(self, vocab2id):
        super(SNLI, self).__init__()

        self.embedding = UnweightedDME(glove_path=GLOVE_PATH, fast_text_path=FAST_TEXT_PATH,
                                       levy_dep_path=LEVY_DEP_PATH, vocab2id=vocab2id)
        self.encoder = Encoder(EMBEDDING_PROJECTION, LSTM_DIM)
        self.classifier = MLP(2 * 4 * LSTM_DIM, MLP_HIDDEN_LAYER, DROPOUT, TAGSET_SIZE)

        print('model size: {:,}'.format(count_param_num(self)))

    def forward(self, premise, hypothesis):
        u = self.embedding(premise[0])
        u = self.encoder(u, premise[1])
        v = self.embedding(hypothesis[0])
        v = self.encoder(v, hypothesis[1])

        m = torch.cat([u, v, (u - v).abs(), u * v], dim=1)
        output = self.classifier(m)
        return output


if __name__ == '__main__':
    model = SNLI()
    hypothesis = 'Two women are sitting on a blanket near some rocks talking about politics.'
    premise = 'Two women are wandering along the shore drinking iced tea.'

    out = model(hypothesis, premise)
