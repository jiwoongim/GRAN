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


import os
import sys

import theano
import theano.tensor as T

import numpy as np
import scipy as sp

from batch_norm_conv_layer import *
from batch_norm_layer import *
from conv_layer import *
from deconv import *
from utils import *

import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState()
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))

class RecGenI28(object):

    def __init__(self, model_params, nkerns=[8,4,1], ckern=32, filter_sizes=[7,5,5]):
        """Initialize the architecture of the model"""
        [self.batch_sz, num_dims, self.num_hids, numpy_rng,\
                self.dim_sample, nkerns, ckern, self.num_channels, self.num_steps]  = model_params 
        self.nkerns     = np.asarray(nkerns) * ckern # of constant gen filters in first conv layer
        self.nkerns[-1] = self.num_channels

        self.D =  int(np.sqrt(num_dims / self.nkerns[-1]))
        self.numpy_rng  = numpy_rng
        self.filter_sizes=filter_sizes

        self.L0a = Batch_Norm_layer(self.dim_sample, self.dim_sample, 'W_h_z', numpy_rng)
        self.L1  = Batch_Norm_layer(self.dim_sample*2, self.num_hids[0], 'W_h_hf', numpy_rng)
        self.L0b = Batch_Norm_layer(self.num_hids[0], self.dim_sample, 'W_z_f', numpy_rng)

        self.L2 = Deconv_layer(self.batch_sz, numpy_rng, tnkern=self.nkerns[1], bnkern=self.nkerns[0] , bfilter_sz=filter_sizes[0], tfilter_sz=filter_sizes[1])
        self.L3 = Deconv_layer(self.batch_sz, numpy_rng, tnkern=self.nkerns[2], bnkern=self.nkerns[1] , bfilter_sz=filter_sizes[1], tfilter_sz=filter_sizes[2])

        self.L5 = BN_Conv_layer(self.batch_sz, numpy_rng, tnkern=self.nkerns[1], bnkern=self.nkerns[0] , bfilter_sz=filter_sizes[0], tfilter_sz=filter_sizes[1])
        self.L6 = BN_Conv_layer(self.batch_sz, numpy_rng, tnkern=self.nkerns[2], bnkern=self.nkerns[1] , bfilter_sz=filter_sizes[1], tfilter_sz=filter_sizes[2])

        self.params = self.L0a.params + self.L0b.params + self.L1.params +   \
                        self.L2.params + self.L3.params + self.L5.params + self.L6.params


    def init_params(self, numpy_rng):
        '''initialize the weights and biases to be used throughout the model'''

        self.W_z_f     = initialize_weight(self.num_hids[0], self.dim_sample, 'W_z_f', numpy_rng, 'uniform')
        self.z_f_bias  = theano.shared(np.zeros((self.dim_sample,), dtype=theano.config.floatX), name='z_f_bias')

        self.W_h_z  = initialize_weight(self.dim_sample, self.dim_sample, 'W_h_z', numpy_rng, 'uniform')
        self.h_z_bias  = theano.shared(np.zeros((self.dim_sample,), dtype=theano.config.floatX), name='h_z_bias')

        self.W_h_hf  = initialize_weight(self.dim_sample*2, self.num_hids[0], 'W_h_hf', numpy_rng, 'uniform')
        self.h_hf_bias  = theano.shared(np.zeros((self.num_hids[0],), dtype=theano.config.floatX), name='h_hf_bias')

        self.param0 = [self.W_h_z, self.h_z_bias, self.W_z_f, self.z_f_bias, self.W_h_hf, self.h_hf_bias]


    def forward(self, Z, H_Ct, testF=False):
        """
        The forward propagation to generate the C_i image.
        """
        H0 = self.L0a.propagate(Z, testF=testF, atype='tanh')
        R0 = T.concatenate([H0.T, H_Ct.T]).T

        H1 = self.L1.propagate( R0, testF=testF, atype='relu')
        H1 = H1.reshape((R0.shape[0], self.nkerns[0], self.filter_sizes[0], self.filter_sizes[0]))
        H2 = self.L2.deconv(H1, atype='relu')
        H3 = self.L3.deconv(H2, atype='linear') 

        return H3

    
    def backward(self,Ct, atype='relu', testF=False):
        """
        backprop or "inverse" stage to update h_i
        """
        H0 = self.L6.conv(Ct, atype=atype)
        H1 = self.L5.conv(H0, atype=atype)
        H1 = H1.flatten(2)
        
        return self.L0b.propagate(H1, testF=testF, atype='tanh')


    def get_samples(self, num_sam, scanF=True):
        """
        Retrieves the samples for the current time step. 
        uncomment parts when time step changes.
        """
        print 'Get_sample func: Number of steps iterate over ::: %d' % self.num_steps

        H_Ct    = T.alloc(0., num_sam, self.dim_sample)
        Zs      = MRG.normal(size=(self.num_steps, num_sam, self.dim_sample), avg=0., std=1.)

        Canvases = self.apply_recurrence(self.num_steps, Zs, H_Ct)
        C        = T.sum(T.stacklists(Canvases),axis=0)

        return activation_fn_th(C, atype='sigmoid'), Canvases


    def apply_recurrence(self, num_steps, Zs, H_Ct, scanF=True):
        """ 
        Applies the recurrent architecture of Model described in GRAN paper, to recursively 
        input previous step's encoded Canvas(H_Ct) with random noise(Z), and receive the 
        encoded H_Ct for the current time step
        It returns the final output Canvas, C.
        """
        def recurrence(i, H_Ct, Zs):

            Z    = Zs[0]
            Ct   = self.forward(Z, H_Ct) 
            H_Ct = self.backward(Ct) 
            return H_Ct, Ct

        if scanF:
            [H_Ct, C], updates = theano.scan(fn=recurrence, outputs_info=[H_Ct, None], \
                        sequences=[T.arange(num_steps)], non_sequences=[Zs])
        else:
            for i in xrange(num_steps):
                H_Ct, C = recurrence(i, H_Ct, C, Zs)
        
        return C


    def weight_decay(self):
        """l2 weight decay used in the optimize_gan for computing the cost of the discriminator"""
        return 0.5*((self.L2.W **2).sum() \
                                    + (self.L3.W**2).sum() + (self.L4.W**2).sum())


