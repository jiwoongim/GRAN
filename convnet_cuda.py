import theano
import theano.tensor as T
import theano.tensor.slinalg as Tkron
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.sandbox.cuda.dnn import dnn_conv
import numpy as np

from conv_layer import *
from batch_norm_conv_layer import *
import os
import sys
from utils import *

class convnet():

    def __init__(self, model_params, nkerns=[1,8,4,2], ckern=128, filter_sizes=[5,5,5,5,4]):

        self.num_hid, num_dims, num_class, self.batch_size, self.num_channels = model_params
        self.D =  int(np.sqrt(num_dims / self.num_channels))
        numpy_rng=np.random.RandomState(1234)

        self.nkerns     = np.asarray(nkerns) * ckern # of constant gen filters in first conv layer
        self.nkerns[0] = self.num_channels
        self.filter_sizes=filter_sizes

        num_convH = self.nkerns[-1]*filter_sizes[-1]*filter_sizes[-1]

        self.W      = initialize_weight(num_convH,  self.num_hid,  'W', numpy_rng, 'uniform') 
        self.hbias  = theano.shared(np.zeros((self.num_hid,), dtype=theano.config.floatX), name='hbias_enc')       
        self.W_y    = initialize_weight(self.num_hid, num_class,  'W_y', numpy_rng, 'uniform') 
        #self.ybias  = theano.shared(np.zeros((num_class,), dtype=theano.config.floatX), name='ybias')

        self.L1 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[0], bnkern=self.nkerns[1] , bfilter_sz=filter_sizes[0], tfilter_sz=filter_sizes[1])
        self.L2 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[1], bnkern=self.nkerns[2] , bfilter_sz=filter_sizes[1], tfilter_sz=filter_sizes[2])
        self.L3 = BN_Conv_layer(self.batch_size, numpy_rng, tnkern=self.nkerns[2], bnkern=self.nkerns[3] , bfilter_sz=filter_sizes[2], tfilter_sz=filter_sizes[3])

        self.num_classes = num_class
        self.params = [self.W_y, self.W, self.hbias] + self.L1.params + self.L2.params + self.L3.params 


    def propagate(self, X, num_train=None, atype='relu'):

        image_shape0=[X.shape[0], self.num_channels, self.D, self.D]
        ConX = X.reshape(image_shape0)
        H0 = self.L1.conv(ConX, atype=atype)
        H1 = self.L2.conv(H0, atype=atype)
        H2 = self.L3.conv(H1, atype=atype) 
        H2 = H2.flatten(2)

        #H3 = activation_fn_th(T.dot(H2, self.W) + self.hbias, atype='relu')
        H3 = activation_fn_th(T.dot(H2, self.W) + self.hbias, atype='tanh')
        y  = T.nnet.sigmoid(T.dot(H3, self.W_y))    

        return y


    def cost(self, X, y):
        p_y_x = self.propagate(X)
        return -T.mean(T.log(p_y_x)[T.arange(y.shape[0]), y])
   
   
    def weight_decay_l2(self):
        #return 0.5 * (T.sum(self.W**2))
        return 0.5 * (T.sum(self.W**2)+T.sum(self.W_y**2))


    def weight_decay_l1(self):
        return T.sum(abs(self.W)) 


    def errors(self, X, y, num_train=None):

        p_y_x   = self.propagate(X, num_train=num_train).flatten()
        pred_y  = p_y_x  > 0.5
        #pred_y  = T.argmax(p_y_x, axis=1)
        return T.mean(T.neq(pred_y, y))


    def set_params(self, params):

        [self.W, self.hbias, self.W_y, self.ybias, self.W0, self.b0, self.W1, self.b1] = params
        self.params = params


def get_training_fns(model, train_set, valid_set, test_set, hyper_params):

    [num_dims, num_train_cases,num_valid_cases, num_test_cases, num_hid, num_class, \
                num_epoch, epsilon, batch_size, model_type, lam, num_channels] \
                                                                = hyper_params

    X = T.fmatrix('X')
    y = T.ivector('y'); 
    lr = T.fscalar('lr')
    index = T.lscalar()  # index to a [mini]batch

    # the cost we minimize during training is the NLL of the model
    cost_train = model.cost(X,y) #+ lam * model.weight_decay() 
    cost_valid = model.cost(X,y)
    grads = T.grad(cost_train, model.params)    


    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - grad_i * T.cast(lr, dtype=theano.config.floatX))
        for param_i, grad_i in zip(model.params, grads)
    ] 

    train_model = theano.function(
        [index, theano.Param(lr,default=epsilon)],
        cost_train,
        updates=updates,
        givens={
            X : train_set[0][index * batch_size: (index + 1) * batch_size],
            y : train_set[1][index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function([index],
        cost_valid,
        givens={X:valid_set[0][index * batch_size: (index + 1) * batch_size],
                y:valid_set[1][index * batch_size: (index + 1) * batch_size]}
    )

    # create a function to compute the mistakes that are made by the model
    valid_err = theano.function([index],
        model.errors(X,y),
        givens={X:valid_set[0][index * batch_size: (index + 1) * batch_size],
                y:valid_set[1][index * batch_size: (index + 1) * batch_size]})
 

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index],
        model.errors(X,y),
        givens={X:test_set[0][index * batch_size: (index + 1) * batch_size],
                y:test_set[1][index * batch_size: (index + 1) * batch_size]})
 
    return train_model, test_model, validate_model, valid_err


