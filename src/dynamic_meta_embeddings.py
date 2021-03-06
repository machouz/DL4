from __future__ import unicode_literals
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math

EMBEDDING_PROJECTION = 256
GLOVE_DIM = 300
FAST_TEXT_DIM = 300
LEVY_DIM = 300


def get_embedding(pretrained_embedding_path, embedding_size, vocab2id):
    pretrained_embedding = load_embedding(pretrained_embedding_path)

    embedding = nn.Embedding(num_embeddings=len(vocab2id), embedding_dim=embedding_size)
    data = embedding.weight.data
    for word, id in vocab2id.items():
        if word in pretrained_embedding:
            data[id] = pretrained_embedding[word] - pretrained_embedding[word].mean(0)
        elif word.lower() in pretrained_embedding:
            data[id] = pretrained_embedding[word.lower()] - pretrained_embedding[word.lower()].mean(0)
        else:
            data[id] = 0

    return embedding


class UnweightedDME(nn.Module):
    def __init__(self, glove_path, fast_text_path, levy_dep_path, vocab2id, dropout=0.2):
        super(UnweightedDME, self).__init__()
        self.vocab2id = vocab2id

        self.glove = get_embedding(glove_path, GLOVE_DIM, self.vocab2id)
        self.fast_text = get_embedding(fast_text_path, FAST_TEXT_DIM, self.vocab2id)
        self.levy = get_embedding(levy_dep_path, LEVY_DIM, self.vocab2id)

        self.P_glove = nn.Linear(GLOVE_DIM, EMBEDDING_PROJECTION)
        self.P_fast_text = nn.Linear(FAST_TEXT_DIM, EMBEDDING_PROJECTION)
        self.P_levy = nn.Linear(LEVY_DIM, EMBEDDING_PROJECTION)

        bound = math.sqrt(3.0 / GLOVE_DIM)
        self.glove.weight.data.uniform_(-1.0 * bound, bound)

        bound = math.sqrt(3.0 / FAST_TEXT_DIM)
        self.fast_text.weight.data.uniform_(-1.0 * bound, bound)

        bound = math.sqrt(3.0 / LEVY_DIM)
        self.levy.weight.data.uniform_(-1.0 * bound, bound)

        torch.nn.init.xavier_uniform_(self.P_glove.weight)
        torch.nn.init.xavier_uniform_(self.P_fast_text.weight)
        torch.nn.init.xavier_uniform_(self.P_levy.weight)

        self.glove.weight.requires_grad = False
        self.fast_text.weight.requires_grad = False
        self.levy.weight.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        self.dropout.cuda()
        if torch.cuda.is_available():
            print("Cuda available")
            self.glove.cuda()
            self.fast_text.cuda()
            self.levy.cuda()
            self.P_glove.cuda()
            self.P_fast_text.cuda()
            self.P_levy.cuda()
            self.dropout.cuda()

    def forward(self, ids):
        emb_glove = self.glove(ids)
        emb_fast_text = self.fast_text(ids)
        emb_levy = self.levy(ids)

        out_glove = self.P_glove(emb_glove)
        out_fast_text = self.P_fast_text(emb_fast_text)
        out_levy = self.P_levy(emb_levy)

        stacked = torch.stack([out_glove, out_fast_text, out_levy])

        output = torch.sum(stacked, dim=0)
        output = F.relu(output)
        output = self.dropout(output)
        return output


if __name__ == '__main__':
    glove_path = 'cache/matched_glove.pkl'
    fast_text_path = 'cache/matched_crawl.pkl'
    vocab_path = 'cache/vocab.pkl'

    unweighted = UnweightedDME(glove_path, fast_text_path, vocab_path)
