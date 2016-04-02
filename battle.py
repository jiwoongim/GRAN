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


import os, sys, gzip, time, timeit
import theano 
import numpy as np
import scipy as sp

import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState()
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimize_gan import *
from gran import *
from utils import * 


datapath='/data/lisa/data/cifar10/cifar-10-batches-py/'
#datapath='/eecs/research/asr/chris/DG_project/dataset/cifar-10-batches-py/'


''' Battle between two models M1 and M2'''
def battle(model1, model2, test_data, num_sam=1000, D=32, num_channel=3):

    #Generate samples from the two models
    samples1 = model1.get_samples(num_sam).reshape((num_sam, D*D*num_channel))
    samples2 = model2.get_samples(num_sam).reshape((num_sam, D*D*num_channel))   

    #Set the target
    target1 = T.alloc(1, num_sam)
    target0 = T.alloc(0, num_sam)

 
    err_m1_fake = model1.dis_network.errors(samples2, target0, num_train=num_sam).eval()
    err_m2_fake = model2.dis_network.errors(samples1, target0, num_train=num_sam).eval()

    err_m1_true = model1.dis_network.errors(test_data, target1, num_train=num_sam).eval()
    err_m2_true = model2.dis_network.errors(test_data, target1, num_train=num_sam).eval()


    print 'Model1 Err on True %g | Err on Fake %g' % (err_m1_true, err_m1_fake)
    print 'Model2 Err on True %g | Err on Fake %g' % (err_m2_true, err_m2_fake)

    sample_ratio = (1-err_m1_fake)/(1-err_m2_fake)
    test_ratio   = (1-err_m1_true)/(1-err_m2_true)

    if sample_ratio > 1:
        print '*** Model1 Wins!!! ***'
    else:
        print '*** Model2 Wins!!! ***'
    print 'Test Ratio %g ' % test_ratio
    print 'Sample Ratio %g ' % sample_ratio


def load_model(fname):

    print '...Continuing from Last time'''
    model = unpickle(os.path.dirname(os.path.realpath(__file__)) + '/params/'+fname)
    return model



