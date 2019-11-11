import json
from nltk.tokenize import word_tokenize
from collections import defaultdict
import re


def tokenize(x):
    x = re.sub('([\\\'".!?,-/])', r' \1 ', x)
    x = re.sub('(\d+)', r' \1 ', x)

    x = word_tokenize(x.lower())

    return x


UNK_TOKEN = "unk"
BATCH_SIZE = 16


def get_frequency_token_vocab(list_tokenized_sentences, vocab=defaultdict(int)):
    for sentence in list_tokenized_sentences:
        for token in sentence:
            vocab[token] += 1

    vocab[UNK_TOKEN] = 10000
    return vocab


def get_mapping_dict(vocab, cutoff=10):
    i = 1
    word_freq = [(k, v) for k, v in vocab.items()]
    word_freq.sort(key = lambda x: -x[1])
    mapping = {}
    for token, freq in word_freq:
        if vocab[token] >= cutoff:
            mapping[token] = i
            i += 1
    return mapping


train = json.load(open("input/filtred_train_data.json", 'r'))
val = json.load(open("input/filtred_val_data.json", 'r'))

list_images_train, captions_train = list(zip(*train))

list_images_val, captions_val = list(zip(*val))

list_tok = [tokenize(x) for x in captions_train+captions_val]

vocab = get_frequency_token_vocab(list_tok)
mapping = get_mapping_dict(vocab)

json.dump(mapping, open('mapping.json', 'w'), indent=4)