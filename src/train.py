from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from SNLI import SNLI
from torchtext import data
from utils import *

SEED = 1234
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0004
LR_DECAY = 0.5

VOCAB_PATH = 'checkpoints/cache/vocab.pkl'
ROOT_PATH = sys.argv[1] if len(sys.argv) > 1 else 'data/datasets/snli/snli_1.0'
TRAIN_PATH = sys.argv[2] if len(sys.argv) > 2 else 'snli_1.0_train.tokenized.prep.json'
VAL_PATH = sys.argv[3] if len(sys.argv) > 2 else 'snli_1.0_dev.tokenized.prep.json'
TEST_PATH = sys.argv[4] if len(sys.argv) > 2 else 'snli_1.0_test.tokenized.prep.json'
VOCAB_EMBED_PATH = 'checkpoints/cache/vocab-fasttext_glove_levy.pkl'


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def get_tensor(sentence, vocab2id):
    ten = torch.tensor([vocab2id[word] for word in sentence])
    if torch.cuda.is_available():
        ten = ten.cuda()
    return ten


def get_data(root_path, train_path, val_path, test_path):
    text_field = data.Field(include_lengths=True, init_token='<s>', eos_token='</s>')
    label_field = data.Field(sequential=False)

    field = {'label': ('label', label_field), 'sentence1': ('premise', text_field),
             'sentence2': ('hypothesis', text_field)}

    train, val, test = data.TabularDataset.splits(path=root_path, train=train_path,
                                                  validation=val_path, test=test_path, format='json', fields=field)

    text_field.build_vocab(train, val, test, min_freq=1)
    return train, val, test, text_field


def get_batchs(model, train, val, test):
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=BATCH_SIZE,
                                                                 repeat=False, sort=False)

    train_iter.shuffle = True
    val_iter.shuffle = False
    test_iter.shuffle = False

    return train_iter, val_iter, test_iter


def train(model, train_iter, dev_iter, test_iter):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for id, batch in enumerate(train_iter):
            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, batch.label - 1)
            optimizer.step()


def evaluate(data_iter, model, type):
    correct = count = 0.0
    model.eval()
    for batch in data_iter:
        optimizer.zero_grad()
        output = model(batch.hypothesis, batch.premise)
        target = batch.label - 1
        correct_prediction = (torch.max(output, 1)[1].view(batch.label.size()).data == target).sum()
        correct += float(correct_prediction.item())
        '''
        if torch.cuda.is_available():
            correct += float(pred.item())
        else:
            correct += (pred == target).cpu().sum().item()
            
        '''
        count += batch.label.shape[-1]

    acc = correct / count
    print("{} - Accuracy on the {}: {}".format(time.strftime('%x %X'), type, acc))




if __name__ == '__main__':
    print("SEED : {}".format(SEED))
    print("BATCH_SIZE : {}".format(BATCH_SIZE))
    print("EPOCHS : {}".format(EPOCHS))
    print("LEARNING_RATE : {}".format(LEARNING_RATE))
    print("VOCAB_PATH : {}".format(VOCAB_PATH))
    print("ROOT_PATH : {}".format(ROOT_PATH))
    print("TRAIN_PATH : {}".format(TRAIN_PATH))
    print("VAL_PATH : {}".format(VAL_PATH))
    print("TEST_PATH : {}".format(TEST_PATH))
    # train, val, test, vocab = get_data(ROOT_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH)
    # train_iter, val_iter, test_iter = get_batchs(train, val, test)

    embedded_vocab = get_embeds_vocab(VOCAB_EMBED_PATH)

    def filter_by_emb_vocab(x):
        return [w for w in x if get_emb_key(w, embedded_vocab)[0] is not None]


    text_field = data.Field(include_lengths=True, init_token='<s>', eos_token='</s>', preprocessing=filter_by_emb_vocab)
    label_field = data.Field(sequential=False)

    field = {'label': ('label', label_field), 'sentence1': ('premise', text_field),
             'sentence2': ('hypothesis', text_field)}

    train, val, test = data.TabularDataset.splits(path=ROOT_PATH, train=TRAIN_PATH,
                                                  validation=VAL_PATH, test=TEST_PATH, format='json', fields=field)

    text_field.build_vocab(train, val, test)
    label_field.build_vocab(train)
    # vocab2id = get_vocab2id(VOCAB_PATH)


    if torch.cuda.is_available():
        train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=BATCH_SIZE,
                                                                     device=torch.device(0), repeat=False, sort=False)

    else:
        train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=BATCH_SIZE,
                                                                     repeat=False, sort=False)

    train_iter.shuffle = True
    val_iter.shuffle = False
    test_iter.shuffle = False


    model = SNLI(text_field.vocab.stoi)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        print("Epoch : {}".format(epoch))
        batch_loss = 0
        for id, batch in enumerate(train_iter):
            optimizer.zero_grad()
            preds = model(batch.hypothesis, batch.premise)
            target = batch.label - 1
            if torch.cuda.is_available():
                target = target.cuda()
            loss = criterion(preds, target)
            batch_loss += float(loss.item())
            loss.backward()
            optimizer.step()
            if id % 100 == 0 and id > 0:
                print("{} - Average loss : {} - id : {}".format(time.strftime('%x %X'), batch_loss / BATCH_SIZE,
                                                                id * BATCH_SIZE))
                batch_loss = 0


        evaluate(train_iter, model, 'train')
        evaluate(val_iter, model, 'dev')
        evaluate(test_iter, model, 'test')
