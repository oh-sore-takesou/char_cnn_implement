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
from data_helper import get_batch, get_datasets, load_datasets, get_un_shuffled_datasets
import slack_api

def train():
    model = CharCNN()
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    bs = 50
    epoch = 100
    ds = 10000

    # datasets = get_datasets(ds) # max: 190864
    # train = datasets[:int(len(datasets)*0.8)]
    # test = datasets[-int(len(datasets)*0.2):]
    p_ds, n_ds = get_un_shuffled_datasets(ds=ds/2)
    train = np.concatenate((p_ds[:int(len(p_ds)*0.8)],  n_ds[:int(len(n_ds)*0.8)]))
    test = np.concatenate((p_ds[-int(len(p_ds)*0.2):], n_ds[-int(len(n_ds)*0.2):]))
    N = len(train)
    x_train = np.array([d[0] for d in train])
    y_train = np.array([d[1] for d in train])

    slack_api.send('here we go')
    slack_api.send('data size: {}'.format(ds))
    slack_api.send('batch size: {}'.format(bs))
    slack_api.send('epoch: {}'.format(epoch))
    for i in range(epoch):
        loss_sum = 0
        eval_correct = 0
        perm = np.random.permutation(N)
        for j in tqdm(range(0, len(train), bs)):
            if i % 3 == 0:
                optimizer.lr = optimizer.lr * 0.5
            X = Variable(np.asarray(x_train[perm[j:j+bs]]))
            Y = Variable(np.asarray(y_train[perm[j:j+bs]]))
            model.cleargrads()
            # loss = model(X, Y)
            YT = model.fwd(X)
            for yt, y in zip(YT, Y):
                if np.argmax(yt.data) == np.argmax(y.data):
                    eval_correct += 1
            loss = F.mean_squared_error(YT, Y)
            loss_sum += float(loss.data) * len(X)
            loss.backward()
            optimizer.update()
        print('{} epoch done loss sum: {}'.format(i+1, loss_sum))
        print('{}% correct when train epoch {} done'.format(eval_correct/len(train)*100, i+1))
        slack_api.send('{} epoch done loss sum: {}'.format(i+1, loss_sum))
        slack_api.send('{}% correct when train epoch {} done'.format(eval_correct/len(train)*100, i+1))

        # eval
        eval_correct = 0
        for j in tqdm(range(0, len(test), bs)):
            X = Variable(np.array([d[0] for d in test[j:j+bs]]))
            Y = np.array([d[1] for d in test[j:j+bs]])
            YT = model.fwd(X, train=False)
            for y, yt in zip(Y, YT):
                y = np.array(y)
                if np.argmax(y) == np.argmax(yt.data):
                    eval_correct += 1
        print(eval_correct)

        print('{}% correct when epoch {} done'.format(eval_correct/len(test)*100, i+1))
        slack_api.send('{}% correct when epoch {} done'.format(eval_correct/len(test)*100, i+1))

        # with open('./m.pkl', 'wb') as f:
        #     pickle.dump(model, f)

if __name__ == '__main__':
    train()
