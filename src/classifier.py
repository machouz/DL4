from __future__ import unicode_literals

import torch.nn as nn
from torch.nn.utils.rnn import *


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class MLP(nn.Module):
    def __init__(self, input_layer, hidden_layer, dropout, tagset_size):
        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_layer, hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layer, tagset_size),
        )

        self.classifier.apply(init_weights)

        if torch.cuda.is_available():
            print("Cuda available")
            self.classifier.cuda()

    def forward(self, encoded, batch=True):
        return self.classifier(encoded)
