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
from util_cifar10 import *

#datapath='/data/lisa/data/cifar10/cifar-10-batches-py/'
#datapath='/eecs/research/asr/chris/DG_project/dataset/cifar-10-batches-py/'
datapath='/home/daniel/Documents/data/cifar10/cifar-10-batches-py/'

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


num_sam=1000
if __name__ == '__main__':

    train_set, valid_set, test_set = load_cifar10(path=datapath)
    train_set[0] = train_set[0] / 255.
    valid_set[0] = valid_set[0] / 255.
    test_set[0]  = test_set[0]  / 255.
    test_set = [test_set[0][:num_sam,:], test_set[1][:num_sam]]
    N ,D = train_set[0].shape; Nv,D = valid_set[0].shape; Nt,D = test_set[0].shape
    print("test_set[1].shape: ", test_set[1].shape)
    print(test_set[1])
    test_set  = shared_dataset(test_set )

    #fname1 = 'dcgan_param_cifar10_2.save'
    fname1 = 'gran_param_cifar10_ts5_3.save'
    fname2 = 'gran_param_cifar10_ts7_2.save'
    #fname2 = 'recgan_gran_param_cifar10_ts5.save5.save'
    fname3 = 'gran_param_cifar10_ts9_2.save'
    model1 = load_model(fname1)
    model2 = load_model(fname2)
    model3 = load_model(fname3)

    print 'gran5 vs gran7'
    battle(model1,model2, test_set[0], Nt) 
    print 'gran5 vs gran9'
    battle(model1,model3, test_set[0], Nt) 
    print 'gran7 vs gran9'
    battle(model2,model3, test_set[0], Nt) 


