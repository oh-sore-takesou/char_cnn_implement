import json
from ast import literal_eval
import numpy as np
from tqdm import tqdm
from sys import exit
import random
import time

def load_datas(filepath):
    datasets = None
    with open(filepath, 'r') as f:
        datas = f.readlines()
    x_positive = []
    x_negative = []
    for data in tqdm(datas):
        dataset = literal_eval(data)
        if dataset['overall'] <= 2:
            x_negative.append(dataset['reviewText'])
        elif dataset['overall'] >= 4:
            x_positive.append(dataset['reviewText'])

    return [ x_positive, x_negative ]

def make_datasets():
    datasets = {}
    x_positives, x_negatives = load_datas('./reviews_Electronics_5.json')
    datasets['x_positives'] = x_positives
    datasets['x_negatives'] = x_negatives
    with open('./datasets.json', 'w') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)


def load_datasets(filepath='./datasets.json'):
    with open(filepath, 'r') as f:
        d = json.load(f)

    return [ d['x_positives'], d['x_negatives'] ]

def convert_token(token, vocab_dict):
    try:
        token_id = vocab_dict[token]
        vec = np.zeros(len(vocab_dict))
        vec[token_id-1] = 1
    except:
        vec = np.zeros(len(vocab_dict))

    return vec.astype(np.float32)

# token -> one hot vec
def convert_tokens(tokens, sen_max=1024):
    with open('./vocab_dict.json', 'r') as f:
        vocab_dict = json.load(f)
    vecs = np.array([convert_token(token, vocab_dict) for token in tokens])
    if len(tokens) == 0:
        vecs = np.zeros(69*sen_max).reshape(sen_max, 69)
    if len(vecs) < sen_max:
        vecs = np.concatenate((vecs, np.zeros(69*(sen_max-len(vecs))).reshape(sen_max-len(vecs), 69)))
    else:
        vecs = vecs[:sen_max]
    return vecs.reshape(1, 1024, 69).astype(np.float32)

def get_batch(bs=10, ds=100):
    positives, negatives = load_datasets()
    positive_sets = np.array([ tuple([convert_tokens(positive), np.array([1,0]).astype(np.float32)]) for positive in positives[:ds] ])
    negative_sets = np.array([ tuple([convert_tokens(negative), np.array([0,1]).astype(np.float32)]) for negative in negatives[:ds] ])
    datasets = np.concatenate((positive_sets, negative_sets))
    np.random.shuffle(datasets)
    for i in range(0, len(datasets), bs):
        yield datasets[i:i+bs]

def get_datasets(ds=1000):
    positives, negatives = load_datasets()
    positive_sets = np.array([ tuple([convert_tokens(positive), np.array([1,0]).astype(np.float32)]) for positive in positives[:ds] ])
    negative_sets = np.array([ tuple([convert_tokens(negative), np.array([0,1]).astype(np.float32)]) for negative in negatives[:ds] ])
    datasets = np.concatenate((positive_sets, negative_sets))
    np.random.shuffle(datasets)

    return datasets


if __name__ == '__main__':
    for batch in get_batch(bs=100, ds=1000):
        print(batch)
