from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.nn.utils.rnn import *


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm):
        super(Encoder, self).__init__()

        self.hidden_lstm = hidden_lstm
        self.lstm = nn.LSTM(embedding_dim, hidden_lstm, bidirectional=True)
        self.init_hidden()

        # torch.nn.init.orthogonal_(self.lstm)
        if torch.cuda.is_available():
            print("Cuda available")
            self.lstm.cuda()

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        self.hidden = (torch.randn(2, batch_size, self.hidden_lstm),
                       torch.randn(2, batch_size, self.hidden_lstm))

    def detach_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def forward(self, embedded, length):
        seq_lens, idx_sort = torch.sort(length, descending=True)
        _, idx_unsort = torch.sort(idx_sort, descending=False)

        sorted_input = embedded.index_select(1, torch.autograd.Variable(idx_sort))
        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input, seq_lens.tolist())
        packed_output = self.lstm(packed_input)[0]
        sorted_rnn_output = nn.utils.rnn.pad_packed_sequence(packed_output)[0]
        rnn_output = sorted_rnn_output.index_select(1, torch.autograd.Variable(idx_unsort))

        rnn_output = rnn_output.max(0)[0]

        return rnn_output
