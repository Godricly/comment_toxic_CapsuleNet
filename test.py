import sys
import os
import argparse
import mxnet as mx
import numpy as np
from preprocess import fetch_test_data
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader,Dataset
from mxnet.io import NDArrayIter
from mxnet.ndarray import array
from mxnet import nd
from net import net_define, net_define_eu
import config

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    # ctx = mx.cpu()# gpu(7)
    ctx = mx.gpu(args.gpu)
    # net = net_define_eu()
    net = net_define()
    net.collect_params().reset_ctx(ctx)
    net.load_params('net0.params', ctx)

    test_data, test_id = fetch_test_data(True)
    data_iter = NDArrayIter(data= test_data, batch_size=args.batch_size, shuffle=False)
    with open('result.txt','w') as f:
        f.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
        for i, d in enumerate(data_iter):
            print(i)
            output=net(d.data[0].as_in_context(ctx)).asnumpy()
            for j in range(args.batch_size):
                if i*args.batch_size + j < test_id.shape[0]:
                    str_out = ','.join([str(test_id[i*args.batch_size+j])] + [str(v) for v in output[j]])+'\n'
                    f.write(str_out)

    
