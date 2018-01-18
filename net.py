import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn,rnn
from conv_cap import PrimeConvCap
from capsule_block import CapFullyBlock, LengthBlock
import config

def net(ctx):
    net = nn.Sequential()
    net.add(nn.Embedding(config.MAX_WORDS, config.MAX_LENGTH))
    net.add(nn.Dropout(0.2))
    net.add(rnn.LSTM(128,layout='NTC',bidirectional=True))
    net.add(transpose(axes=(0,2,1)))
    net.add(PrimeConvCap(8,32, kernel_size=(9,1), padding=(4,0)))
    net.add(CapFullyBlock( 8*config.MAX_LENGTH, num_cap=6, input_units=32, units=16, context=ctx))
    # net.add(LengthBlock())
    net.add(nn.Dense(6, activation='sigmoid'))
    return net

class transpose(nn.Block):
    def __init__(self, axes, **kwargs):
        super(transpose, self).__init__(**kwargs)
        self.axes = axes

    def forward(self, x):
        return nd.transpose(x, axes=self.axes).reshape((0,0,0,1))

