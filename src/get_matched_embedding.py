import sys

from six.moves import cPickle as pickle

EMBEDDING_SIZE = 300

'''
def get_vocabulary(train, dev, test):
    text_field = data.Field(include_lengths=True, init_token='<s>', eos_token='</s>')
    train, dev, test = SNLIDataset.splits(text_field, label_field, args.datasets_root)
    text_field.build_vocab(train, dev, test, min_freq=1)
    return text_field
'''


def get_matched_embeddings(embedding_path, vocab, path):
    with open(embedding_path, 'rb') as f:
        embed = pickle.load(f)

    with open(vocab) as v:
        vocab2id = pickle.load(v)

    new_embed = {}
    for word, id in vocab2id.items():
        if word in embed:
            new_embed[word] = embed[word]
        elif word.lower() in embed:
            new_embed[word.lower()] = embed[word.lower()]

    with open(path, 'wb') as f:
        pickle.dump(new_embed, f, protocol=2)


if __name__ == '__main__':
    embedding_path = sys.argv[1]
    vocab = sys.argv[2]
    path = sys.argv[3]

    get_matched_embeddings(embedding_path, vocab, path)
