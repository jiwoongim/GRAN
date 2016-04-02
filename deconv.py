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

class Deconv_layer(object):
    
    def __init__ (self, batch_sz, numpy_rng, tnkern=5, \
                    bfilter_sz=5, tfilter_sz=5, bnkern=1, poolsize=(2,2)):

        self.filter_shape   =(bnkern, tnkern, tfilter_sz, tfilter_sz) #TODO 
        self.init_conv_filters(numpy_rng, bfilter_sz, poolsize)


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

        b_values = np.zeros((self.filter_shape[1],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name='b_conv')
        self.params = [self.W, self.b]
    

    def deconv(self, X, subsample=(2, 2), border_mode=(2, 2), conv_mode='conv', atype='sigmoid'):
        """ 
        sets up dummy convolutional forward pass and uses its grad as deconv
        currently only tested/working with same padding
        """
    
        #Always return a c contiguous output.
        #Copy the input only if it is not already c contiguous.
        img = gpu_contiguous(X)
        kerns = gpu_contiguous(self.W)
    
        #Implement Alloc on the gpu, but without initializing memory.
        gpu_alloc_img_shape = gpu_alloc_empty(img.shape[0], kerns.shape[1], \
                img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape
    
        #This Op builds a convolution descriptor for use in the other convolution operations.
        desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                            conv_mode=conv_mode)(gpu_alloc_img_shape, kerns.shape)
    
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0],\
                                                    img.shape[3]*subsample[1])
    
        #The convolution gradient with respect to the inputs.
        d_img = GpuDnnConvGradI()(kerns, img, out, desc)
        return activation_fn_th(d_img + self.b.dimshuffle('x', 0, 'x', 'x'), atype=atype) 
  


    def conv(self, X, subsample=(2, 2), border_mode=(2, 2), conv_mode='conv', atype='sigmoid'):

        ConH0 = dnn_conv(X , self.W, subsample=subsample, border_mode=border_mode)
        return activation_fn_th(ConH0 + self.b.dimshuffle('x', 0, 'x', 'x'), atype=atype)



