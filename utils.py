from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
import mxnet as mx

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def accuracy(output, label):
    L = -label*nd.log2(output) - (1-label) * nd.log2(1-output)
    return nd.mean(L).asscalar()

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    data = batch.data[0]
    label = batch.label[0]
    # data, label = gluon.utils.split_and_load(batch, ctx)
    return data.as_in_context(ctx), label.as_in_context(ctx)


def evaluate_accuracy(data_iterator, net, ctx=mx.gpu()):
    acc = 0.
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc / (i+1)


def train(train_data, test_data, net, loss, trainer,
          ctx, num_epochs, print_batches=None):
    """Train a network"""
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
                print("Batch %d. Loss: %f, Train acc %f, Test acc %f" % (
                n, train_loss/n, train_acc/n, test_acc))
        test_acc = evaluate_accuracy(test_data, net, ctx)
        train_data.reset()
        test_data.reset()
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
              epoch, train_loss/n, train_acc/n, test_acc))

