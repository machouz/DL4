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



    def forward(self, encoded, batch=True):
        return self.classifier(encoded)