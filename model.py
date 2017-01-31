import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.initializers import HeNormal

class CharCNN(Chain):

    def __init__(self):
        initializer = HeNormal()
        super(CharCNN, self).__init__(
            conv1=L.Convolution2D(1, out_channels=1, ksize=7, initialW=initializer),
            conv2=L.Convolution2D(1, out_channels=1, ksize=7, initialW=initializer),
            conv3=L.Convolution2D(1, out_channels=1, ksize=7, initialW=initializer),
            conv4=L.Convolution2D(1, out_channels=1, ksize=7, initialW=initializer),
            conv5=L.Convolution2D(1, out_channels=1, ksize=7, initialW=initializer),
            conv6=L.Convolution2D(1, out_channels=1, ksize=7, initialW=initializer),
            l1=L.Linear(None, 2048),
            l2=L.Linear(None, 2048),
            l3=L.Linear(None, 2),
        )

    def __call__(self, x, y, train=True):
        return F.mean_squared_error(self.fwd(x, train=train), y)

    def fwd(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), (1, 3))
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), (1, 3))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.relu(self.conv5(h4))
        h6 = F.max_pooling_2d(F.relu(self.conv6(h5)), (1, 3))
        h7 = F.dropout(F.relu(self.l1(h6)), ratio=0.5, train=train)
        h8 = F.dropout(F.relu(self.l2(h7)), ratio=0.5, train=train)
        y = self.l3(h8)
        return y


class TestLinear(Chain):

    def __init__(self):
        super(TestLinear, self).__init__(
            l1=L.Linear(None, 2048),
            l2=L.Linear(None, 2048),
            l3=L.Linear(None, 10),
        )

    def __call__(self, x, y, train=True):
        return F.mean_squared_error(self.fwd(x, train=train), y)

    def fwd(self, x, train=True):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return h3
