import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet import initializer



def squash(x, axis):
    s_squared_norm = nd.sum(nd.square(x), axis, keepdims=True)
    # if s_squared_norm is really small, we will be in trouble
    # so I removed the s_quare terms
    # scale = s_squared_norm / ((1 + s_squared_norm) * nd.sqrt(s_squared_norm + 1e-9))
    # return x * scale
    scale = nd.sqrt(s_squared_norm + 1e-9)
    return x / scale


class CapConvBlock(nn.Block):
    def __init__(self, num_cap, channels, context, kernel_size=(9,9), padding=(0,0),
                 strides=(1,1), route_num=3, **kwargs):
        super(CapConvBlock, self).__init__(**kwargs)
        self.num_cap = num_cap
        self.cap = nn.Conv2D(channels=channels*num_cap, kernel_size=kernel_size,
                                  strides=strides, padding=padding)
        self.route_num = route_num

    def forward(self, x):
        conv_out = nd.expand_dims(self.cap(x), axis=2)
        conv_out = conv_out.reshape((0,-1,self.num_cap,0,0))
        conv_out  = squash(conv_out, 1)
        return conv_out
        
class CapFullyBlock(nn.Block):
    def __init__(self, num_locations, num_cap, input_units, units,
                 route_num=3, **kwargs):
        super(CapFullyBlock, self).__init__(**kwargs)
        self.route_num = route_num
        self.num_cap = num_cap
        self.units = units
        self.num_locations = num_locations
        self.w_ij = self.params.get(
             'weight', shape=(input_units, units, self.num_cap, self.num_locations)
             ,init=init.Xavier()) 

    def forward(self, x):
        # reshape x into [batch_size, channel, num_previous_cap]
        x_reshape = nd.transpose(x,(0,2,1,3,4)).reshape((0,0,-1))
        return self.Route(x_reshape)

    def Route(self, x):
        # b_mat = nd.repeat(self.b_mat.data(), repeats=x.shape[0], axis=0)#nd.stop_gradient(nd.repeat(self.b_mat.data(), repeats=x.shape[0], axis=0))
        b_mat = nd.zeros((x.shape[0],1,self.num_cap, self.num_locations), ctx=x.context)
        x_expand = nd.expand_dims(nd.expand_dims(x, axis=2),2)
        w_expand = nd.repeat(nd.expand_dims(self.w_ij.data(x.context),axis=0), repeats=x.shape[0], axis=0)
        u_ = w_expand*x_expand
        # u_ = nd.abs(w_expand - x_expand)
        u = nd.sum(u_, axis = 1)
        u_no_gradient = nd.stop_gradient(u)
        for i in range(self.route_num):
            c_mat = nd.softmax(b_mat, axis=2)
            if i == self.route_num -1:
                s = nd.sum(u * c_mat, axis=-1)
            else:
                s = nd.sum(u_no_gradient * c_mat, axis=-1)
            v = squash(s, 1)
            v1 = nd.expand_dims(v, axis=-1)
            if i != self.route_num - 1:
                update_term = nd.sum(u_no_gradient*v1, axis=1, keepdims=True)
                b_mat = b_mat + update_term
        return v


class CapFullyNGBlock(nn.Block):
    def __init__(self, num_locations, num_cap, input_units, units,
                 route_num=3, **kwargs):
        super(CapFullyNGBlock, self).__init__(**kwargs)
        self.route_num = route_num
        self.num_cap = num_cap
        self.units = units
        self.num_locations = num_locations
        self.w_ij = self.params.get(
             'weight', shape=(input_units, units, self.num_cap, self.num_locations)
             ,init=init.Xavier()) 

    def forward(self, x):
        # reshape x into [batch_size, channel, num_previous_cap]
        x_reshape = nd.transpose(x,(0,2,1,3,4)).reshape((0,0,-1))
        return self.Route(x_reshape)

    def Route(self, x):
        # b_mat = nd.repeat(self.b_mat.data(), repeats=x.shape[0], axis=0)#nd.stop_gradient(nd.repeat(self.b_mat.data(), repeats=x.shape[0], axis=0))
        b_mat = nd.zeros((x.shape[0],1,self.num_cap, self.num_locations), ctx=x.context)
        x_expand = nd.expand_dims(nd.expand_dims(x, axis=2),2)
        w_expand = nd.repeat(nd.expand_dims(self.w_ij.data(x.context),axis=0), repeats=x.shape[0], axis=0)
        u_ = w_expand*x_expand
        u = nd.sum(u_, axis = 1)
        for i in range(self.route_num):
            c_mat = nd.softmax(b_mat, axis=2)
            s = nd.sum(u * c_mat, axis=-1)
            v = squash(s, 1)
            v1 = nd.expand_dims(v, axis=-1)
            update_term = nd.sum(u * v1, axis=1, keepdims=True)
            b_mat = b_mat + update_term
        return v


class CapFullyEuBlock(nn.Block):
    def __init__(self, num_locations, num_cap, input_units, units,
                 route_num=3, **kwargs):
        super(CapFullyEuBlock, self).__init__(**kwargs)
        self.route_num = route_num
        self.num_cap = num_cap
        self.units = units
        self.num_locations = num_locations
        self.w_ij = self.params.get(
             'weight', shape=(input_units, units, self.num_cap, self.num_locations)
             ,init=init.Xavier()) 

    def forward(self, x):
        # reshape x into [batch_size, channel, num_previous_cap]
        # print x.shape
        x_reshape = nd.transpose(x,(0,2,1,3,4)).reshape((0,0,-1))
        return self.Route(x_reshape)

    def Route(self, x):
        # print x.context
        # b_mat = nd.repeat(self.b_mat.data(), repeats=x.shape[0], axis=0)#nd.stop_gradient(nd.repeat(self.b_mat.data(), repeats=x.shape[0], axis=0))
        b_mat = nd.zeros((x.shape[0],1,self.num_cap, self.num_locations), ctx=x.context)
        x_expand = nd.expand_dims(nd.expand_dims(x, axis=2),2)
        w_expand = nd.repeat(nd.expand_dims(self.w_ij.data(x.context),axis=0), repeats=x.shape[0], axis=0)
        u_ = w_expand*x_expand
        u = nd.sum(u_, axis = 1)
        # u_ = nd.square(w_expand - x_expand)
        # u = -nd.sum(u_, axis = 1)
        u_no_gradient = nd.stop_gradient(u)
        for i in range(self.route_num):
            # c_mat = nd.softmax(b_mat, axis=2)
            c_mat = nd.sigmoid(b_mat)
            if i == self.route_num -1:
                s = nd.sum(u * c_mat, axis=-1)
            else:
                s = nd.sum(u_no_gradient * c_mat, axis=-1)
            v = squash(s, 1)
            if i != self.route_num - 1:
                v1 = nd.expand_dims(v, axis=-1)
                update_term = nd.sum(u_no_gradient*v1, axis=1, keepdims=True)
                b_mat = b_mat + update_term
                # b_mat = update_term
            # else:
            #    v = s
        return v

class LengthBlock(nn.Block):
    def __init__(self, **kwargs):
        super(LengthBlock, self).__init__(**kwargs)

    def forward(self, x):
        x = nd.sqrt(nd.sum(nd.square(x), 1))
        return x

