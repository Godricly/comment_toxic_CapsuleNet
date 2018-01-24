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
from net import net_define
import config

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()

    # ctx = mx.cpu()# gpu(7)

    for i in range(args.kfold):
        ctx = mx.gpu(args.gpu)
        net = net_define()
        net.collect_params().reset_ctx(ctx)
        net.load_params('net'+str(i)+'.params', ctx)
        test_data, test_id = fetch_test_data()
        # test_data = test_data[:20]
        data_iter = NDArrayIter(data= test_data, batch_size=1, shuffle=False)
        with open('result'+str(i)+'.csv','w') as f:
            f.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
            for i, d in enumerate(data_iter):
                print (i)
                output=net(d.data[0].as_in_context(ctx))
                str_out = ','.join([str(test_id[i])] + [str(v) for v in output[0].asnumpy()])+'\n'
                f.write(str_out)
    
