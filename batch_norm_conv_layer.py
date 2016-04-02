''' Version 1.000
 Code provided by Daniel Jiwoong Im and Chris Dongjoo Kim
 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

'''Demo of Generating images with recurrent adversarial networks.
For more information, see: http://arxiv.org/abs/1602.05110
'''

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import *
TINY    = 1e-6

class BN_Conv_layer(object):
    
    def __init__ (self, batch_sz, numpy_rng, tnkern=5, \
                    bfilter_sz=5, tfilter_sz=5, bnkern=1, poolsize=(2,2)):
        """Parameter Initialization for Batch Norm"""

        self.filter_shape   =(tnkern, bnkern, tfilter_sz, tfilter_sz) #TODO 

        self.eta         = theano.shared(np.ones((bnkern,), dtype=theano.config.floatX), name='eta') 
        self.beta        = theano.shared(np.zeros((bnkern,), dtype=theano.config.floatX), name='beta')
        self.stat_mean   = theano.shared(np.zeros((bnkern,), dtype=theano.config.floatX), name='running_avg')
        self.stat_std    = theano.shared(np.zeros((bnkern,), dtype=theano.config.floatX), name='running_std')

        self.init_conv_filters(numpy_rng, bfilter_sz, poolsize)
        self.params += [self.eta, self.beta]


    def init_conv_filters(self, numpy_rng, D, poolsize):

        ''' Convolutional Filters '''
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(self.filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" pooling size
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) /
                   np.prod(poolsize))

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(
                init_conv_weights(-W_bound, W_bound, \
                        self.filter_shape, numpy_rng),borrow=True, name='W_conv')


        self.params = [self.W]


    def collect_statistics(self, X):
        """Updates Statistics of data"""
        stat_mean = T.mean(X, axis=0)
        stat_std  = T.std(X, axis=0)

        updates_stats = [(self.stat_mean, stat_mean), (self.stat_std, stat_std)]
        return updates_stats


    def conv(self, X, subsample=(2, 2), border_mode=(2, 2), atype='sigmoid', testF=False):

        ConH0 = dnn_conv(X , self.W.dimshuffle(1,0,2,3), subsample=subsample, border_mode=border_mode)
        if testF:
            ConH1 = (ConH0 - self.stat_mean.dimshuffle('x', 0, 'x', 'x')) \
                                / (self.stat_std.dimshuffle('x', 0, 'x', 'x') + TINY) 
        else:
            mean    = ConH0.mean(axis=[0,2,3]).dimshuffle('x', 0, 'x', 'x')
            std     = ConH0.std( axis=[0,2,3]).dimshuffle('x', 0, 'x', 'x')
            ConH1   = (ConH0 - mean) / (std + TINY)
    
        ConH2 = self.eta.dimshuffle('x', 0, 'x', 'x') * ConH1 \
                                    + self.beta.dimshuffle('x', 0, 'x', 'x')

        return activation_fn_th(ConH2, atype=atype)


