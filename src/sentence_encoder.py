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



class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm):
        super(Encoder, self).__init__()

        self.hidden_lstm = hidden_lstm
        self.lstm = nn.LSTM(embedding_dim, hidden_lstm, bidirectional=True, batch_first=True)

        self.init_hidden()

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (torch.randn(2, batch_size, self.hidden_lstm),
                        torch.randn(2, batch_size, self.hidden_lstm))


    def detach_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def forward(self, embedded, batch=False):
        if batch:
            lstm_out, self.hidden = self.lstm(embedded)
            encoded_sentence = PackedSequence(
                lstm_out.data.max(), lstm_out.batch_sizes)
        else:
            embedded = embedded.unsqueeze(0)
            lstm_out, self.hidden = self.lstm(embedded)
            lstm_out = lstm_out.reshape(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))

            encoded_sentence = lstm_out.max(0)[0] # max return max , idx


        return encoded_sentence
