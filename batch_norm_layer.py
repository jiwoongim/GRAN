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


import os, sys
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.sandbox.rng_mrg as RNG_MRG

from utils import *
TINY    = 1e-6


class Batch_Norm_layer(object):

    def __init__(self, D, M, name, numpy_rng):
        """Parameter Initialization for Batch Norm"""
        self.W       = initialize_weight(D, M,  name, numpy_rng, 'uniform') 
        self.zbias       = theano.shared(np.zeros((M,), dtype=theano.config.floatX), name='zbias')
        self.eta         = theano.shared(np.ones((M,), dtype=theano.config.floatX), name='eta') 
        self.beta        = theano.shared(np.zeros((M,), dtype=theano.config.floatX), name='beta')
        self.stat_mean   = theano.shared(np.zeros((M,), dtype=theano.config.floatX), name='running_avg')
        self.stat_std    = theano.shared(np.zeros((M,), dtype=theano.config.floatX), name='running_std')

        self.params = [self.W, self.zbias, self.eta, self.beta]

    def collect_statistics(self, X):
        """ updates statistics on data"""
        stat_mean = T.mean(X, axis=0)
        stat_std  = T.std(X, axis=0)

        updates_stats = [(self.stat_mean, stat_mean), (self.stat_std, stat_std)]
        return updates_stats


    def propagate(self, X, testF=False, atype='sigmoid'):

        H = self.pre_activation(X, testF=False)
        H = activation_fn_th(H, atype=atype)
        return H

    def pre_activation(self, X, testF=False):

        Z = self.post_batch_norm(X, testF=testF)
        H = self.eta * Z + self.beta
        return H

    def post_batch_norm(self, X, testF=False):

        Z = T.dot(X, self.W) + self.zbias   
        if testF:
            Z       = (Z - self.stat_mean) / (self.stat_std + TINY)
        else:
            mean    = Z.mean(axis=0)
            std     = Z.std( axis=0)
            Z       = (Z - mean) / (std + TINY)

        return Z

