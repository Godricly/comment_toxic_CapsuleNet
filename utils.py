from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from sklearn.metrics import roc_auc_score, confusion_matrix
import mxnet as mx
import numpy as np
import time

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def accuracy(output, label):
    L = -label*np.log2(output) - (1-label) * np.log2(1-output)
    return np.mean(L)

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    data = batch.data[0]
    label = batch.label[0]
    # data, label = gluon.utils.split_and_load(batch, ctx)
    return data.as_in_context(ctx), label.as_in_context(ctx)

def _get_batch_multi(batch, ctx, Train=True):
    # naive random shuffle
    if Train:
        npdata = batch.data[0].asnumpy()
        np_roll_data = np.roll(npdata, axis=1, shift=np.random.randint(npdata.shape[1]))
        nd_data = nd.array(np_roll_data )
        data = gluon.utils.split_and_load(nd_data, ctx)
    else:
        data = gluon.utils.split_and_load(batch.data[0], ctx)
    label = gluon.utils.split_and_load(batch.label[0], ctx)
    return data, label

def evaluate_accuracy(data_iterator, net, ctx=mx.gpu()):
    acc = 0.
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc / (i+1)

def evaluate_accuracy_multi(data_iterator, net, ctx):
    data_iterator.reset()
    acc = 0
    dummy_label = np.zeros((0,6))
    dummy_pred = np.zeros((0,6))
    t1 = time.time()
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch_multi(batch, ctx, False)
        # acc += np.mean([accuracy(net(X), Y) for X, Y in zip(data, label)])
        # acc += np.mean([roc_auc_score(Y.asnumpy(), net(X).asnumpy()) for X, Y in zip(data, label)])
        output = np.vstack((net(X).asnumpy() for X in data))
        labels = np.vstack((Y.asnumpy() for Y in label))
        dummy_label = np.vstack((dummy_label, labels)) 
        dummy_pred = np.vstack((dummy_pred, output))
    # return acc / (i+1)
    # print dummy_label.shape, dummy_pred.shape
    dummy_pred_label = dummy_pred > 0.5
    for i in range(dummy_label.shape[1]):
        print i, confusion_matrix(dummy_label[:,i], dummy_pred_label[:,i])

    return roc_auc_score(dummy_label, dummy_pred), accuracy(dummy_pred, dummy_label), time.time() - t1


def train(train_data, test_data, net, loss, trainer,
          ctx, num_epochs, print_batches=None):
    """Train a network"""
    min_loss = 100000
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        n = 0
        for i, batch in enumerate(train_data):
            data, label = _get_batch(batch, ctx)
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            trainer.step(data.shape[0], ignore_stale_grad=True)
            train_loss += nd.mean(L).asscalar()
            train_acc += accuracy(output, label)
            n = i + 1
            if print_batches and n % print_batches == 0:
                test_acc = evaluate_accuracy(test_data, net, ctx)
                test_data.reset()
                print("Batch %d. Loss: %f, Train acc %f, Test Loss %f" % (
                n, train_loss/n, train_acc/n, test_acc))
                if test_acc < min_loss:
                    min_loss = test_acc
                    net.save_params('net.params')
        test_acc = evaluate_accuracy(test_data, net, ctx)
        train_data.reset()
        test_data.reset()
        print("Epoch %d. Loss: %f, Train acc %f, Test Loss %f" % (
              epoch, train_loss/n, train_acc/n, test_acc))
        if test_acc < min_loss:
            min_loss = test_acc
            net.save_params('net.params')

def train_multi(train_data, test_data, iteration, net, loss, trainer,
          ctx, num_epochs, print_batches=None, pos_tr_ratio=None):
    """Train a network"""
    min_loss = 0
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        n = 0
        for i, batch in enumerate(train_data):
            data, label = _get_batch_multi(batch, ctx)
            with autograd.record():
                losses = [loss(net(X), Y, pos_tr_ratio) for X, Y in zip(data, label)]
                for l in losses:
                    l.backward()
            trainer.step(batch.data[0].shape[0], ignore_stale_grad=True)
            train_loss += np.mean([nd.mean(l).asscalar() for l in losses])
            # train_acc += accuracy(output, label)
            n = i + 1
            if print_batches and n % print_batches == 0:
                test_acc, test_loss, eval_time = evaluate_accuracy_multi(test_data, net, ctx)
                print("Batch %d. Loss: %f, Test roc_auc: %f, test_loss: %f , eval time: %f" % (
                n, train_loss/n, test_acc, test_loss, eval_time))
                if test_acc > min_loss:
                    min_loss = test_acc
                    net.save_params('net'+str(iteration)+'.params')
          
        train_data.reset()
        test_acc, test_loss, eval_time = evaluate_accuracy_multi(test_data, net, ctx)
        print("Epoch %d. Loss: %f, roc_auc: %f, test_loss: %f , eval time: %f" % (
              epoch, train_loss/n, test_acc, test_loss, eval_time))
        if test_acc > min_loss:
            min_loss = test_acc
            net.save_params('net'+str(iteration)+'.params')

