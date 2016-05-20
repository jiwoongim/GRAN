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


import time, timeit
import hickle as hkl
import theano 
import numpy as np
import scipy as sp
import os, sys, glob
import gzip

import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState()
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimize_gan import *
# from recGanI import *
from gran import *
from deconv import *
from utils import * 
from util_cifar10 import * 

debug = sys.gettrace() is not None
if debug:
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'
    theano.config.compute_test_value = 'warn'
    
#datapath='/export/mlrg/imj/machine_learning/data/lsun/cifar-10-batches-py/'
#datapath='/u/imdaniel/Documents/machine_learning/collaborate/gan/data/lsun/preprocessed_toy/0000.hkl'
# datapath='/eecs/research/asr/chris/DG_project/dataset/lsun/preprocessed_toy_100/'
datapath = '/local/scratch/chris/church/preprocessed_toy_100/'

if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/figs/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/figs/")
if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/figs/lsun"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/figs/lsun")
if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/params/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/params/")


def lets_train(model, train_params, num_batchs, theano_fns, opt_params, model_params):

    ganI_params, conv_params = model_params 
    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params   
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps= ganI_params
    num_epoch, epoch_start, contF, train_filenames, valid_filenames, test_filenames = train_params 
    num_batch_train, num_batch_valid, num_batch_test                        = num_batchs
    get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost = theano_fns

    print '...Start Training'
    findex= str(num_hids[0])+'_'
    best_vl = np.infty    
    K=1 #FIXED
    num_samples =100;
    for epoch in xrange(num_epoch+1):

        costs=[[],[], []]
        exec_start = timeit.default_timer()

        eps_gen = get_epsilon(epsilon_gen, 25, epoch)
        eps_dis = get_epsilon(epsilon_dis, 25, epoch)
        for batch_i in xrange(num_batch_train):
           
            data = hkl.load(train_filenames[batch_i]) / 255.
            data = data.astype('float32').transpose([3,0,1,2]);
            a,b,c,d = data.shape
            data = data.reshape(a,b*c*d)
            cost_disc_i = discriminator_update(data, lr=eps_dis)
            costs[0].append(cost_disc_i)
            
            if batch_i % K == 0:
                cost_gen_i = generator_update(lr=eps_gen)
                costs[1].append(cost_gen_i)

        exec_finish = timeit.default_timer() 
        if epoch==0: print 'Exec Time %f ' % ( exec_finish - exec_start)


        if epoch < 6 or epoch > 2 or epoch == (num_epoch-1):

            costs_vl = [[],[],[]]
            for batch_j in xrange(num_batch_valid):
                data = hkl.load(valid_filenames[batch_j]) / 255.
                data = data.astype('float32').transpose([3,0,1,2]);
                a,b,c,d = data.shape
                data = data.reshape(a, b*c*d)
                cost_dis_vl_j, cost_gan_vl_j = get_valid_cost(data)
                costs_vl[0].append(cost_dis_vl_j)
                costs_vl[1].append(cost_gan_vl_j)
            #    print("validation success !");

            cost_dis_vl = np.mean(np.asarray(costs_vl[0]))
            cost_gan_vl = np.mean(np.asarray(costs_vl[1]))             

            cost_dis_tr = np.mean(np.asarray(costs[0]))
            cost_gan_tr = np.mean(np.asarray(costs[1]))

            cost_tr = cost_dis_tr+cost_gan_tr
            cost_vl = cost_dis_vl+cost_gan_vl

            print 'Epoch %d, epsilon_gen %f5, epsilon_dis %f5, tr disc gen %g, %g | vl disc gen %g, %g '\
                    % (epoch, eps_gen, eps_dis, cost_dis_tr, cost_gan_tr, cost_dis_vl, cost_gan_vl)

            num_samples=100
            samples = get_samples(num_samples).reshape((num_samples, 64*64*3))
            display_images(np.asarray(samples * 255, dtype='int32'), tile_shape = (10,10), img_shape=(64,64), fname='./figs/lsun/RG2/1_gan_samples500_' + model_param_save + str(epoch));

            # change the name to save to when new model is found.
            save_the_weight(model, './params/recgan_'+ model_param_save + str(epoch))# + findex+ str(K)) 

    num_samples=100
    samples = get_samples(num_samples).reshape((num_samples, 3*64*64))
    display_images(np.asarray(samples * 255, dtype='int32'), tile_shape=(10,10), img_shape=(64,64), fname='./figs/lsun/RG1/1_gan_samples500_'+ '_'+ findex + str(K))

    return model


def load_model(model_params, contF=True):

    if not contF:
        print '...Starting from the beginning'''
        model = GRAN(model_params)
    else:
        print '...Continuing from Last time'''
        path_name = raw_input("Enter full path to the pre-trained model: ")
        model = unpickle(path_name)

    return model 


def set_up_train(model, opt_params):

    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt, lam = opt_params
    opt_params    = batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt
    compile_start = timeit.default_timer()
    opt           = Optimize(opt_params)

    print ("Compiling...it may take a few minutes")
    discriminator_update, generator_update, get_valid_cost, get_test_cost\
                    = opt.optimize_gan_hkl(model)
    get_samples     = opt.get_samples(model)
    compile_finish = timeit.default_timer() 
    print 'Compile Time %f ' % ( compile_finish - compile_start) 
    return opt, get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost


def main(opt_params, ganI_params, train_params, conv_params):

    batch_sz, epsilon_gen, epsilon_dis,  momentum, num_epoch, N, Nv, Nt, lam    = opt_params  
    batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps    = ganI_params 
    conv_num_hid, D, num_class, batch_sz, num_channel                           = conv_params  
    num_epoch, epoch_start, contF,train_filenames, valid_filenames, test_filenames  = train_params 
    num_batch_train = len(train_filenames)
    num_batch_valid = len(valid_filenames)
    num_batch_test  = len(test_filenames)

    model_params = [ganI_params, conv_params]
    ganI = load_model(model_params, contF)
    opt, get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost\
                                = set_up_train(ganI, opt_params)

    #TODO: If you want to train your own model, comment out below section and set the model parameters below accordingly
    ##################################################################################################
    num_samples=100
    fname='./figs/lsun/gran_lsun_samples500.pdf'
    samples = get_samples(num_samples).reshape((num_samples, 3*64*64))
    display_images(np.asarray(samples * 255, dtype='int32'), tile_shape=(10,10), img_shape=(64,64),fname=fname)
    print ("LSUN sample fetched and saved to " + fname)
    exit()
    ###################################################################################################
    theano_fns = [get_samples, discriminator_update, generator_update, get_valid_cost, get_test_cost]
    num_batchs = [num_batch_train, num_batch_valid, num_batch_test]
    lets_train(ganI, train_params, num_batchs, theano_fns, opt_params, model_params)


### MODEL PARAMS
# CONV (DISC)
conv_num_hid= 100
num_channel = 3 # FIXED
num_class   = 1 # FIXED
D           = 64*64*3

# ganI (GEN)
filter_sz   = 4 #FIXED
nkerns      = [1,8,4,2,1]
ckern       = 172
num_hid1    = nkerns[0]*ckern*filter_sz*filter_sz # FIXED.
num_steps   = 5 # time steps
num_z       = 100 

### OPT PARAMS
batch_sz    = 100
epsilon_dis = 0.0001
epsilon_gen = 0.0002
momentum    = 0.0 #Not Used
lam1        = 0.000001 

### TRAIN PARAMS
num_epoch   = 15
epoch_start = 0 
contF       = False #continue flag. usually FIXED
N=1000 
Nv=N 
Nt=N #Dummy variable
D = 12288

### SAVE PARAM
model_param_save = 'num_hid%d.batch%d.eps_dis%g.eps_gen%g.num_z%d.num_epoch%g.lam%g.ts%d.data.100_CONV_lsun'%(conv_num_hid,batch_sz, epsilon_dis, epsilon_gen, num_z, num_epoch, lam1, num_steps)
#model_param_save = 'gran_param_lsun_ts%d.save' % num_steps

if __name__ == '__main__':
    
    # store the filenames into a list.
    train_filenames = sorted(glob.glob(datapath + 'train_hkl_b100_b_100/*' + '.hkl'))
    valid_filenames = sorted(glob.glob(datapath + 'val_hkl_b100_b_100/*' + '.hkl'))
    test_filenames = sorted(glob.glob(datapath + 'test_hkl_b100_b_100/*' + '.hkl'))

    print 'num_hid%d.batch sz %d, epsilon_gen %g, epsilon_disc %g, num_z %d,  num_epoch %d, lambda %g, ckern %d' % \
                                    (conv_num_hid, batch_sz, epsilon_gen, epsilon_dis, num_z, num_epoch, lam1, ckern)
    num_hids     = [num_hid1]
    train_params = [num_epoch, epoch_start, contF, train_filenames, valid_filenames, test_filenames]
    opt_params   = [batch_sz, epsilon_gen, epsilon_dis,  momentum, num_epoch, N, Nv, Nt, lam1]    
    ganI_params  = [batch_sz, D, num_hids, rng, num_z, nkerns, ckern, num_channel, num_steps]
    conv_params  = [conv_num_hid, D, num_class, batch_sz, num_channel]
    book_keeping = main(opt_params, ganI_params, train_params, conv_params)


