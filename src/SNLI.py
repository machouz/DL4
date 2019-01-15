from __future__ import unicode_literals

from classifier import *
from dynamic_meta_embeddings import *
from sentence_encoder import *

VOCAB_PATH = 'checkpoints/cache/vocab.pkl'
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

    def __init__(self, vocab2id):
        super(SNLI, self).__init__()

        self.embedding = UnweightedDME(GLOVE_PATH, FAST_TEXT_PATH, vocab2id)
        self.encoder = Encoder(EMBEDDING_PROJECTION, LSTM_DIM)
        self.classifier = MLP(2 * 4 * LSTM_DIM, MLP_HIDDEN_LAYER, DROPOUT, TAGSET_SIZE)

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
