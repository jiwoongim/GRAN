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



import os, sys, time, timeit, gzip
import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import theano 
import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState()
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))

from optimize_gan import *
from gran import *
from utils import * 
from util_cifar10 import *

#datapath='/groups/branson/home/imd/Documents/machine_learning_uofg/data/cifar10/cifar-10-batches-py/'
#datapath='/export/mlrg/imj/machine_learning/data/cifar10/cifar-10-batches-py/'
#datapath='/data/lisa/data/cifar10/cifar-10-batches-py/'
#datapath='/eecs/research/asr/chris/DG_project/dataset/cifar-10-batches-py/'
#datapath='/home/imj/data/cifair10/cifar-10-batches-py/'
datapath='/home/daniel/Documents/data/cifar10/cifar-10-batches-py/'

if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/figs/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/figs/")
if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/figs/cifar10"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/figs/cifar10")
if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/params/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/params/")


def lets_train(model, train_params, num_batchs, theano_fns, opt_params, model_params):

    ganI_params, conv_params = model_params 
    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params   
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps= ganI_params
    num_epoch, epoch_start, contF                                           = train_params 
    num_batch_train, num_batch_valid, num_batch_test                        = num_batchs
    get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost = theano_fns

    print '...Start Training'
    findex= str(num_hids[0])+'_'
    best_vl = np.infty    
    K=1 #FIXED
    for epoch in xrange(num_epoch+1):

        costs=[[],[], []]
        exec_start = timeit.default_timer()

        eps_gen = get_epsilon(epsilon_gen, 50, epoch)
        eps_dis = get_epsilon(epsilon_dis, 50, epoch)
        for batch_i in xrange(num_batch_train):
            
            cost_disc_i = discriminator_update(batch_i, lr=eps_dis)
            costs[0].append(cost_disc_i)

            if batch_i % K == 0:
                cost_gen_i = generator_update(lr=eps_gen)
                costs[1].append(cost_gen_i)

        exec_finish = timeit.default_timer() 
        if epoch==0: print 'Exec Time %f ' % ( exec_finish - exec_start)

        if epoch % 5 == 0 or epoch < 4 or epoch == (num_epoch-1):

            costs_vl = [[],[],[]]
            for batch_j in xrange(num_batch_valid):
                cost_dis_vl_j, cost_gan_vl_j = get_valid_cost(batch_j)
                costs_vl[0].append(cost_dis_vl_j)
                costs_vl[1].append(cost_gan_vl_j)

            cost_dis_vl = np.mean(np.asarray(costs_vl[0]))
            cost_gan_vl = np.mean(np.asarray(costs_vl[1]))             

            cost_dis_tr = np.mean(np.asarray(costs[0]))
            cost_gan_tr = np.mean(np.asarray(costs[1]))

            print 'Epoch %d, epsilon_dis %f5, epsilon_gen %f5, tr disc gen %g, %g | vl disc gen %g, %g '\
                    % (epoch, eps_dis, eps_gen, cost_dis_tr, cost_gan_tr, cost_dis_vl, cost_gan_vl)

            num_samples=100
            samples = get_samples(num_samples).reshape((num_samples, 32*32*3))
            display_images(np.asarray(samples * 255, dtype='int32'), (10,10), fname='./figs/cifar10/granI_samples_'+str(epoch) +"_"+ 'ns'+str(num_steps))

            # change the name to save to when new model is found.
            save_the_weight(model, './params/'+ model_param_save )
            
    num_samples=100
    samples = get_samples(num_samples).reshape((num_samples, 3*32*32))
    display_images(np.asarray(samples * 255, dtype='int32'), (10,10), fname='./figs/cifar10/gran_samples_'+ '_'+ findex + 'ns'+str(num_steps))

    return model


def load_model(model_params, contF=True):

    if not contF:
        print '...Starting from the beginning'''
        model = GRAN(model_params)
    else:
        print '...Continuing from Last time'''
        model = unpickle(os.path.dirname(os.path.realpath(__file__)) + '/params/'+'recgan_batch100.eps_dis4e-05.eps_gen7e-05.num_z150.num_epoch70.lam2e-05_cifar103.save')
    return model 


def set_up_train(model, train_set, valid_set, test_set, opt_params):

    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params  
    opt_params    = batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt 
    compile_start = timeit.default_timer()
    opt           = Optimize(opt_params)

    discriminator_update, generator_update, get_valid_cost, get_test_cost\
                    = opt.optimize_gan(model, train_set, valid_set, test_set, lam1=lam)
    get_samples     = opt.get_samples(model)
    compile_finish = timeit.default_timer() 
    print 'Compile Time %f ' % ( compile_finish - compile_start) 

    return opt, get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost


def main(train_set, valid_set, test_set, opt_params, ganI_params, train_params, conv_params):

    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam     = opt_params  
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps    = ganI_params 
    conv_num_hid, D, num_class, batch_sz, num_channel                           = conv_params  
    num_epoch, epoch_start, contF                                               = train_params 

    # compute number of minibatches for training, validation and testing
    num_batch_train = N  / batch_sz
    num_batch_valid = Nv / batch_sz
    num_batch_test  = Nt / batch_sz

    model_params = [ganI_params, conv_params]
    ganI = load_model(model_params, contF)
    opt, get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost\
                                = set_up_train(ganI, train_set, valid_set, test_set, opt_params)


    theano_fns = [get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost]
    num_batchs = [num_batch_train, num_batch_valid, num_batch_test]
    lets_train(ganI, train_params, num_batchs, theano_fns, opt_params, model_params)




### MODEL PARAMS

# CONV (DISC)
conv_num_hid= 100
num_channel = 3 # FIXED
num_class   = 1 # FIXED

# ganI (GEN)
filter_sz   = 4 #FIXED
nkerns      = [8,4,2,1]
ckern       = 128
num_hid1    = nkerns[0]*ckern*filter_sz*filter_sz # FIXED.
num_steps   = 5     # time steps
num_z       = 100   #/ num_steps # To match random noise of GAN's. 

### OPT PARAMS
batch_sz    = 100
epsilon_dis = 0.00004
epsilon_gen = 0.0001
momentum    = 0.0 #Not Used
lam         = 0.000005


### TRAIN PARAMS
num_epoch   = 20#23
epoch_start = 0 
contF       = False #Continue flag. usually FIXED

### SAVE PARAM

model_param_save = 'gran_param_cifar10_ts%d_2' % num_steps

if __name__ == '__main__':

    train_set, valid_set, test_set = load_cifar10(path=datapath)
    train_set[0] = train_set[0] / 255.
    valid_set[0] = valid_set[0] / 255.
    test_set[0]  = test_set[0]  / 255.
 
    N ,D = train_set[0].shape; Nv,D = valid_set[0].shape; Nt,D = test_set[0].shape    
    N_test, D_test = train_set[0][100:200].shape

    train_set = shared_dataset(train_set)
    valid_set = shared_dataset(valid_set)
    test_set  = shared_dataset(test_set )

    print 'batch sz %d, epsilon gen %g, epsilon dis %g, hnum_z %d, num_conv_hid %g, num_epoch %di, lam %g, num_steps %d' % \
                                    (batch_sz, epsilon_gen, epsilon_dis, num_z, conv_num_hid, num_epoch, lam, num_steps)

    num_hids     = [num_hid1]
    train_params = [num_epoch, epoch_start, contF]
    opt_params   = [batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam]    
    ganI_params  = [batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps]
    conv_params  = [conv_num_hid, D, num_class, batch_sz, num_channel]
    book_keeping = main(train_set, valid_set, test_set, opt_params, ganI_params, train_params, conv_params)


