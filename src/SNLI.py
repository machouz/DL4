from __future__ import unicode_literals
import torch.nn as nn
from utils import *
import time
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
import sys
import torch

from classifier import *
from dynamic_meta_embeddings import *
from sentence_encoder import *


GLOVE_PATH = 'checkpoints/cache/matched_glove.pkl'
FAST_TEXT_PATH = 'checkpoints/cache/matched_crawl.pkl'
GLOVE_DIM = 300
FAST_TEXT_DIM = 300
EMBEDDING_PROJECTION = 256

LSTM_DIM = 512
MLP_HIDDEN_LAYER = 1024
DROPOUT = 0.2
TAGSET_SIZE = 3



class SNLI(nn.Module):

    def __init__(self):
        super(SNLI, self).__init__()

        self.embedding = UnweightedDME(GLOVE_PATH, FAST_TEXT_PATH)
        self.encoder = Encoder(EMBEDDING_PROJECTION, LSTM_DIM)
        self.classifier = MLP(2 * 4 * LSTM_DIM, MLP_HIDDEN_LAYER, DROPOUT, TAGSET_SIZE)



    def forward(self, hypothesis, premise):
        u = self.embedding(hypothesis)
        u = self.encoder(u)
        v = self.embedding(premise)
        v = self.encoder(v)

        m = torch.cat([u, v, (u - v).abs(), u * v])
        output = self.classifier(m)
        return output


if __name__ == '__main__':

    model = SNLI()
    hypothesis = 'Two women are sitting on a blanket near some rocks talking about politics.'
    premise = 'Two women are wandering along the shore drinking iced tea.'

    out = model(hypothesis, premise)
