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

from model import CharCNN
from data_helper import get_batch

def train():
    model = CharCNN()
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    bs = 2
    ds = 4
    epoch = 100
    # losses = []
    # for i in tqdm(range(epoch)):
    for i in range(epoch):
        loss_sum = 0
        for batch in tqdm(get_batch(bs=bs, ds=ds)):
            if i % 3 == 0:
                optimizer.lr = optimizer.lr * 0.5
            X = Variable(np.array([d[0] for d in batch]))
            Y = Variable(np.array([d[1] for d in batch]))
            model.cleargrads()
            loss = model(X, Y)
            loss.backward()
            optimizer.update()
            loss_sum += float(loss.data) * len(X)
        print(loss_sum)

if __name__ == '__main__':
    train()
