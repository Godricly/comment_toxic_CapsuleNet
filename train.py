import sys
import os
import argparse
sys.path.insert(0,'../incubator-mxnet/python')
import mxnet as mx
import numpy as np
from preprocess import fetch_data
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader,Dataset
from mxnet.io import NDArrayIter
from mxnet.ndarray import array
from mxnet import nd
from net import net
import utils
import config

def CapLoss(y_pred, y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    args = parser.parse_args()
    train_data, train_label = fetch_data()
    '''
    train_data = np.random.randint(0, high=config.MAX_WORDS, size=(10000, config.MAX_LENGTH))
    train_label = np.random.randint(0, high=6, size=(10000, 6)) 
    '''

    data_iter = NDArrayIter(data= train_data[:-1000], label=train_label[:-1000], batch_size=32, shuffle=True)
    val_data_iter = NDArrayIter(data= train_data[-1000:], label=train_label[-1000:], batch_size=32, shuffle=False)

    ctx = mx.cpu()
    net = net(ctx)
    net.initialize(mx.init.Xavier(),ctx=ctx)
    net.collect_params().reset_ctx(ctx)

    print_batches = 1
    trainer = Trainer(net.collect_params(),'adam', {'learning_rate': 0.001})
    utils.train(data_iter, val_data_iter, net, CapLoss,
                trainer, ctx, num_epochs=args.epochs, print_batches=print_batches)
    net.save_params('net.params')
