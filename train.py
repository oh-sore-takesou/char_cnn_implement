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

from model import CharCNN
from data_helper import get_batch, get_datasets
import slack_api

def train():
    model = CharCNN()
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    bs = 10
    epoch = 100
    # datasets = get_datasets(190000) # max: 190864
    datasets = get_datasets(100) # max: 190864
    train = datasets[:int(len(datasets)*0.8)]
    test = datasets[int(len(datasets)*0.2):]
    eval_correct = 0
    eval_incorrect = 0
    for i in range(epoch):
        loss_sum = 0
        for j in tqdm(range(0, len(train), bs)):
            if i % 3 == 0:
                optimizer.lr = optimizer.lr * 0.5
            X = Variable(np.array([d[0] for d in train[j:j+bs]]))
            Y = Variable(np.array([d[1] for d in train[j:j+bs]]))
            model.cleargrads()
            loss = model(X, Y)
            loss.backward()
            optimizer.update()
            loss_sum += float(loss.data) * len(X)
        print('{} epoch done loss sum: {}'.format(i+1, loss_sum))
        slack_api.send('{} epoch done loss sum: {}'.format(i+1, loss_sum))
        # eval
        X = Variable(np.array([d[0] for d in test]))
        Y = [d[1] for d in test]
        YT = model.fwd(X, train=False)
        for y, yt in zip(Y, YT):
            y = np.array(y)
            if np.argmax(y) == np.argmax(yt.data):
                eval_correct += 1
            else:
                eval_incorrect += 1
        print('{}% correct when epoch {} done'.format(eval_correct/(eval_correct+eval_incorrect)*100, i+1))
        slack_api.send('{}% correct when epoch {} done'.format(eval_correct/(eval_correct+eval_incorrect)*100, i+1))
        with open('./m.pkl', 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    train()
