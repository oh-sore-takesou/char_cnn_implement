import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from tqdm import tqdm
from sys import exit
import pickle

from model import TestLinear
from data_helper import get_batch, get_datasets, load_datasets, get_un_shuffled_datasets, convert_tokens
from eprint import s_print

def eval(text):
    with open('./m.pkl', 'rb') as f:
        model = pickle.load(f)    
    sen = convert_tokens(text)
    X = Variable(np.asarray(np.array(sen).astype(np.float32).reshape(1,1,69,1024)))
    result = model.fwd(X)
    print(result.data)
    print(np.argmax(result.data))

if __name__ == '__main__':
    eval("I have had this product 1 month. It was attached to my clock radio on my night stand. No rough handling, simply used it every night for charging & display. It will not charge less than a month from receiving it. Has a no return policy-buyer beware!")
