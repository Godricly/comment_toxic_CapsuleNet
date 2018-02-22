import sys
import os
import argparse
import mxnet as mx
from mxnet import init
import numpy as np
from preprocess import fetch_data, get_word_embedding, get_embed_matrix
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader,Dataset
from mxnet.io import NDArrayIter
from mxnet.ndarray import array
from mxnet import nd
from net import net_define, net_define_eu
from sklearn.model_selection import KFold, StratifiedKFold
import utils
import config

def CapLoss(y_pred, y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))

def EntropyLoss(y_pred, y_true, train_pos_ratio=None):
    L = - y_true*(1-y_pred)**2*nd.log2(y_pred) - (1-y_true) * nd.log2(1-y_pred)*y_pred**2
    return nd.mean(L)

def EntropyLoss1(y_pred, y_true, train_pos_ratio):
    scale = 10
    train_pos_ratio = array(train_pos_ratio, ctx=y_pred.context, dtype=np.float32) * scale
    train_neg_ratio = (scale - train_pos_ratio)
    L = - y_true*nd.log2(y_pred) * train_neg_ratio - (1-y_true) * nd.log2(1-y_pred)*train_pos_ratio 
    return nd.mean(L)

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--kfold', default=10, type=int)
    parser.add_argument('--print_batches', default=100, type=int)
    args = parser.parse_args()

    train_data, train_label, word_index = fetch_data()
    embedding_dict = get_word_embedding()
    em = get_embed_matrix(embedding_dict, word_index)
    em = array(em, ctx=mx.cpu())
    kf_label = np.ones(train_label.shape)
    for i in range(train_label.shape[1]):
        kf_label[:,i] = 2**i
    kf_label = np.sum(kf_label, axis=1)

    ctx = [mx.gpu(0)]
    net = net_define_eu()

    kf = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    for i, (inTr, inTe) in enumerate(kf.split(train_data, kf_label)):
        print('fold: ', i)
        net.collect_params().initialize(init=init.Xavier(), force_reinit=True)
        xtr = train_data[inTr]
        xte = train_data[inTe]
        ytr = train_label[inTr]
        yte = train_label[inTe]
        pos_tr_ratio = np.sum(ytr, axis=0)/float(ytr.shape[0])
        pos_tr_ratio = np.ones(pos_tr_ratio.shape)*0.5
        data_iter =     NDArrayIter(data= xtr, label=ytr, batch_size=args.batch_size, shuffle=True)
        val_data_iter = NDArrayIter(data= xte, label=yte, batch_size=args.batch_size, shuffle=False)

        # print net.collect_params()
        net.collect_params().reset_ctx(ctx)
        net.collect_params()['sequential0_embedding0_weight'].set_data(em)
        net.collect_params()['sequential0_embedding0_weight'].grad_req = 'null'
        # net.collect_params()['sequential'+str(i)+ '_embedding0_weight'].set_data(em)
        # net.collect_params()['sequential'+str(i)+ '_embedding0_weight'].grad_req = 'null'
        trainer = Trainer(net.collect_params(),'adam', {'learning_rate': 0.001})
        # trainer = Trainer(net.collect_params(),'RMSProp', {'learning_rate': 0.01,'clip_weights' : 1})
        utils.train_multi(data_iter, val_data_iter, i, net, EntropyLoss1,
                    trainer, ctx, num_epochs=args.epochs, print_batches=args.print_batches, pos_tr_ratio=pos_tr_ratio)
