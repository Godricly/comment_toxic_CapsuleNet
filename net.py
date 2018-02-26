import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn,rnn
from conv_cap import PrimeConvCap, AdvConvCap
from capsule_block import CapFullyBlock, CapFullyEuBlock, CapFullyNGBlock, LengthBlock, ActBlock
import config

def net_define():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Embedding(config.MAX_WORDS, config.EMBEDDING_DIM))
        net.add(rnn.GRU(128,layout='NTC',bidirectional=True, num_layers=2, dropout=0.2))
        net.add(transpose(axes=(0,2,1)))
        # net.add(nn.MaxPool2D(pool_size=(config.MAX_LENGTH,1)))
        # net.add(nn.Conv2D(128, kernel_size=(101,1), padding=(50,0), groups=128,activation='relu'))
        net.add(PrimeConvCap(8,32, kernel_size=(1,1), padding=(0,0)))
        # net.add(AdvConvCap(8,32,8,32, kernel_size=(1,1), padding=(0,0)))
        net.add(CapFullyBlock(8*(config.MAX_LENGTH)/2, num_cap=12, input_units=32, units=16, route_num=5))
        # net.add(CapFullyBlock(8*(config.MAX_LENGTH-8), num_cap=12, input_units=32, units=16, route_num=5))
        # net.add(CapFullyBlock(8, num_cap=12, input_units=32, units=16, route_num=5))
        net.add(nn.Dropout(0.2))
        # net.add(LengthBlock())
        net.add(nn.Dense(6, activation='sigmoid'))
    net.initialize(init=init.Xavier())
    return net

def net_define_eu():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Embedding(config.MAX_WORDS, config.EMBEDDING_DIM))
        net.add(rnn.GRU(128,layout='NTC',bidirectional=True, num_layers=1, dropout=0.2))
        net.add(transpose(axes=(0,2,1)))
        net.add(nn.GlobalMaxPool1D())
        '''
        net.add(FeatureBlock1())
        '''
        net.add(extendDim(axes=3))
        net.add(PrimeConvCap(16, 32, kernel_size=(1,1), padding=(0,0),strides=(1,1)))
        net.add(CapFullyNGBlock(16, num_cap=12, input_units=32, units=16, route_num=3))
        net.add(nn.Dropout(0.2))
        net.add(nn.Dense(6, activation='sigmoid'))
    net.initialize(init=init.Xavier())
    return net


class extendDim(nn.Block):
    def __init__(self, axes, **kwargs):
        super(extendDim, self).__init__(**kwargs)
        self.axes = axes

    def forward(self, x):
        x1 = nd.expand_dims(x, axis=self.axes)
        return x1

class reduceDim(nn.Block):
    def __init__(self, **kwargs):
        super(reduceDim, self).__init__(**kwargs)

    def forward(self, x):
        x1 = x.reshape((x.shape[0], x.shape[1], -1))
        return x1
 

class transpose(nn.Block):
    def __init__(self, axes, **kwargs):
        super(transpose, self).__init__(**kwargs)
        self.axes = axes

    def forward(self, x):
        return nd.transpose(x, axes=self.axes)# .reshape((0,0,0,1))

class fullyReshape(nn.Block):
    def __init__(self, axes, **kwargs):
        super(fullyReshape, self).__init__(**kwargs)
        self.axes = axes

    def forward(self, x):
        return nd.transpose(x, axes=self.axes).reshape((0,0,0,1,1))

# hard coding feature Block
class FeatureBlock(nn.Block):
    def __init__(self, **kwargs):
        super(FeatureBlock, self).__init__(**kwargs)
        self.gru = rnn.GRU(128,layout='NTC',bidirectional=True, num_layers=1, dropout=0.2)
        self.conv3 = nn.Conv1D(channels=128, kernel_size=5, padding=2, strides=1, activation='relu')
        self.conv5 = nn.Conv1D(channels=128, kernel_size=9, padding=4, strides=1, activation='relu')
        self.conv7 = nn.Conv1D(channels=128, kernel_size=13, padding=6, strides=1, activation='relu')
        self.conv_drop = nn.Dropout(0.2)

    def forward(self, x):
        gru_out = self.gru(x)
        gru_out_t = nd.transpose(gru_out, axes=(0,2,1))

        x_t = nd.transpose(x, axes=(0,2,1))
        conv3_out = self.conv3(x_t)
        conv5_out = self.conv5(x_t)
        conv7_out = self.conv7(x_t)
        conv_out = nd.concat(*[conv3_out, conv5_out, conv7_out], dim=1)
        conv_out = self.conv_drop(conv_out)
        concated_feature = nd.concat(*[gru_out_t, conv_out], dim=1)
        return concated_feature

# hard coding feature1 Block
class FeatureBlock1(nn.Block):
    def __init__(self, **kwargs):
        super(FeatureBlock1, self).__init__(**kwargs)
        self.gru = rnn.GRU(128,layout='NTC',bidirectional=True, num_layers=1, dropout=0.2)
        self.conv3 = nn.Conv1D(channels=128, kernel_size=3, padding=1, strides=1, activation='relu')
        self.conv5 = nn.Conv1D(channels=128, kernel_size=3, padding=1, strides=1, activation='relu')
        self.conv7 = nn.Conv1D(channels=128, kernel_size=3, padding=1, strides=1, activation='relu')
        # self.gru_post_max = nn.MaxPool1D(pool_size=2)
        # self.gru_post_ave = nn.AvgPool1D(pool_size=2)
        self.gru_maxpool = nn.GlobalMaxPool1D()
        self.conv_maxpool = nn.GlobalMaxPool1D()
        '''
        self.gru_avepool = nn.GlobalAvgPool1D()
        self.conv_avepool = nn.GlobalAvgPool1D()
        '''
        self.conv_drop = nn.Dropout(0.5)

    def forward(self, x):
        x_t = nd.transpose(x, axes=(0,2,1))
        conv3_out = self.conv3(x_t)
        conv5_out = self.conv5(conv3_out) + conv3_out
        conv7_out = self.conv7(conv5_out) + conv5_out 
        # conv_out = nd.concat(*[conv3_out, conv5_out, conv7_out], dim=1)
        conv_out = self.conv_drop(conv7_out)
        conv_max_pooled = self.conv_maxpool(conv_out)

        gru_out = self.gru(x)
        gru_out_t = nd.transpose(gru_out, axes=(0,2,1))
        # gru_pooled = nd.transpose(gru_out, axes=(0,2,1))
        # gru_maxpooled = self.gru_post_max(gru_out_t)
        # return gru_maxpooled
        # gru_avepooled = self.gru_post_ave(gru_out_t)
        # gru_pooled = nd.concat(*[gru_maxpooled, gru_avepooled], dim=1)

        # gru_pooled = nd.concat(*[gru_maxpooled, gru_avepooled], dim=1)
        gru_maxpooled = self.gru_maxpool(gru_out_t)
        # gru_avepooled = self.gru_maxpool(gru_out_t)
        # gru_pooled = nd.concat(*[gru_maxpooled, gru_avepooled], dim=1)

        # conv_ave_pooled = self.conv_avepool(conv_out)
        concated_feature = nd.concat(*[gru_maxpooled, conv_max_pooled], dim=1)
        return concated_feature
