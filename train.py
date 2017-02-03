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

import sobamchan_utility
utility = sobamchan_utility.Utility()
import sobamchan_slack
slack = sobamchan_slack.Slack()

def train():
    model = CharCNN()
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    bs = 128
    epoch = 100
    ds = 1000

    channel = None

    token_dict = utility.load_json('./vocab_dict.json')

    datasets = utility.load_json('./datasets.json')
    train, test = utility.separate_datasets(datasets)
    train_each_N = min(len(train['positive']), len(train['negative']))
    train_x = utility.np_float32([utility.np_float32(utility.convert_one_of_m_vector_char(value, token_dict, 1014)).reshape(1,1,1014) for key, values in train.items() for value in values[:train_each_N]])
    train_y = utility.np_int32([0] * train_each_N + [1] * train_each_N)

    test_each_N = min(len(test['positive']), len(test['negative']))
    test_x = utility.np_float32([utility.np_float32(utility.convert_one_of_m_vector_char(value, token_dict, 1014)).reshape(1,1,1014) for key, values in test.items() for value in values[:test_each_N]])
    test_y = utility.np_int32([0] * test_each_N + [1] * test_each_N)

    slack.s_print('here we go', channel)
    slack.s_print('data size: {}'.format(ds), channel)
    slack.s_print('batch size: {}'.format(bs), channel)
    slack.s_print('epoch: {}'.format(epoch), channel)

    N = len(train_x)
    print(N)

    for i in range(epoch):
        slack.s_print('-'*10, channel)
        loss_sum = 0
        eval_correct = 0
        perm = np.random.permutation(N)
        for j in tqdm(range(0, N, bs)):
            if i % 3 == 0:
                optimizer.lr = optimizer.lr * 0.5
            X = Variable(train_x[perm[j:j+bs]])
            Y = Variable(np.array(train_y[perm[j:j+bs]]))
            model.cleargrads()
            YT = model.fwd(X)
            for yt, y in zip(YT, Y):
                if np.argmax(yt.data) == np.argmax(y.data):
                    eval_correct += 1
            loss = F.softmax_cross_entropy(YT, Y)
            loss_sum += float(loss.data) * len(X)
            loss.backward()
            optimizer.update()
        slack.s_print('{} epoch done loss sum: {}'.format(i+1, loss_sum), channel)
        slack.s_print('{}% correct when train epoch {} done'.format(eval_correct/len(train)*100/bs, i+1), channel)

    #     # eval
    #     eval_correct = 0
    #     for j in tqdm(range(0, len(test), bs)):
    #         X = Variable(np.array([d[0] for d in test_x[j:j+bs]]))
    #         Y = np.array([d[1] for d in test_y[j:j+bs]])
    #         YT = model.fwd(X, train=False)
    #         for y, yt in zip(Y, YT):
    #             y = np.array(y)
    #             if np.argmax(y) == np.argmax(yt.data):
    #                 eval_correct += 1

    #     slack.s_print('{}% correct when epoch {} done'.format(eval_correct/len(test)*100, i+1), channel)

    #     with open('./m.pkl', 'wb') as f:
    #         pickle.dump(model, f)

if __name__ == '__main__':
    train()
