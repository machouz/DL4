from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.nn.utils.rnn import *


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
        if torch.cuda.is_available():
            print("Cuda available")
            self.classifier.cuda()


    def forward(self, encoded, batch=True):
        return self.classifier(encoded)
