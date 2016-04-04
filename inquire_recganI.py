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
import os, sys, glob
import gzip

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tempfile import TemporaryFile
from optimize_gan import *
from recGanI import *
from gran import *
from deconv import *
from utils import * 
from util_cifar10 import *


cifar10_datapath='/eecs/research/asr/chris/DG_project/dataset/cifar-10-batches-py/'
lsun_datapath='/local/scratch/chris/church/preprocessed_toy_10/'
mnist_datapath = '/data/mnist/'

if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/figs/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/figs/")
if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/figs/cifar10"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/figs/cifar10")
if not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/params/"):
    os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/params/")


def visualize_knn(train_set, samples, kfilename, k=7):
   
    Ns,D = samples.shape
    distmtx = (dist2hy(samples, train_set[0]))

    min_knn_ind = T.argsort(distmtx,axis=1)[:,:k]
    closest_datas = train_set[0][min_knn_ind].eval()
    tmp     = np.concatenate([samples.reshape(Ns,1,D), np.ones((Ns,1,D))], axis=1)
    output  = np.concatenate([tmp, closest_datas], axis=1).reshape(Ns*(k+2), D)
    if (filename == 'CIFAR10'):
        display_images(output * 255., (Ns,k+2), fname='./figs/cifar10/inq/nn_ts5')
    elif(filename == 'LSUN'):
        display_images(np.asarray(output * 255, dtype='int32'), tile_shape = (Ns, k+2), img_shape=(64,64),fname='./figs/lsun/inq/nn_ts5_lsun');
    elif(filename == 'MNIST'): 
        display_dataset(output, (28,28), (Ns,k+2), i=1, fname='./figs/MNIST/inq/nn_ts5')
    return    

def load_model(filename, model_name):
    if (model_name == ''):
        if (filename == 'CIFAR10'):
            model = unpickle(os.path.dirname(os.path.realpath(__file__)) + '/params/'+'recgan_num_hid100.batch100.eps_dis0.0001.eps_gen0.0002.num_z100.num_epoch15.lam1e-06.ts3.ckern128.data.10_lsun_get_eps(70).hbias_rem.z=zs[0]10.save')
        elif(filename == 'LSUN'):
            model = unpickle(os.path.dirname(os.path.realpath(__file__)) + '/params/'+'recgan_num_hid100.batch100.eps_dis0.0001.eps_gen0.0002.num_z100.num_epoch15.lam1e-06.ts3.ckern128.data.10_lsun_get_eps(70).hbias_rem.z=zs[0]10.save')
            save_the_weight(model.params, './params/lsun_ts3')
            exit()
 
        elif(filename == 'MNIST'): 
            model = unpickle(os.path.dirname(os.path.realpath(__file__)) + '/params/'+'gran_param_cifar10_ts5_2.save')
    else:
        model = unpickle(os.path.dirname(os.path.realpath(__file__)) + '/params/'+model_name)

    return model


def set_up_train(model, opt_params):

    compile_start = timeit.default_timer()
    opt           = Optimize(opt_params)

    get_samples     = opt.get_samples(model)
    compile_finish  = timeit.default_timer() 
    print 'Compile Time %f ' % ( compile_finish - compile_start) 

    #return opt, get_samples, get_seq_drawing
    return opt, get_samples


def main(train_set, valid_set, test_set, opt_params, filename):

    batch_sz, epsilon_gen, epsilon_dis, momentum, num_epoch, N, Nv, Nt       = opt_params # TODO coonsider making epsilon into epsilon dis and gen separately.  

    # compute number of minibatches for training, validation and testing
    num_batch_train = N  / batch_sz
    num_batch_valid = Nv / batch_sz
    num_batch_test  = Nt / batch_sz

    model = load_model(filename, model_name)
    #opt, get_samples, get_seq_drawing = set_up_train(ganI, train_set, valid_set, test_set, opt_params)

    opt, get_samples = set_up_train(model, opt_params)

    #Flags
    vis_knnF=1
    vis_seqF=1

    if vis_knnF: 
        num_sam=7
        samples = get_samples(num_sam)
        samples = samples.reshape((num_sam, samples.shape[2]*samples.shape[3]*samples.shape[1]))
        knn_samples = visualize_knn(train_set, samples, filename);
    
    if vis_seqF:
        get_seq_drawing = opt.get_seq_drawing(model)
        seq_samples     = get_seq_drawing(10)
        seq_samples     = seq_samples.reshape((seq_samples.shape[0]*seq_samples.shape[1]\
                        ,seq_samples.shape[2]*seq_samples.shape[3]*seq_samples.shape[4]))
        if (filename == 'CIFAR10'):
            display_images(seq_samples * 255., (model.num_steps,10), fname='./figs/cifar10/inq/seq_drawing_ts5_cifar10')
        elif(filename == 'LSUN'):
            display_images(seq_samples * 255., (model.num_steps,10), img_shape=(64,64), fname='./figs/lsun/inq/seq_drawing_ts5_lsun')
        elif(filename == 'MNIST'):
            display_dataset(seq_samples, (28,28), (model.num_steps,10), i=1, fname='./figs/MNIST/inq/seq_drawing_ts5_mnist')

### MODEL PARAMS

# CONV (DISC)
conv_num_hid= 100
num_channel = 3
num_class   = 1

# ganI (GEN)
filter_sz   = 4 #FIXED
nkerns      = [8,4,2,1]
ckern       = 128
num_hid1    = nkerns[0]*ckern*filter_sz*filter_sz
num_z       = 100
lam         = 0.00003

### OPT PARAMS
batch_sz    = 100#128
epsilon     = 0.0002
momentum    = 0.0 #Not Used

### TRAIN PARAMS
num_epoch   = 50
epoch_start = 0 


if __name__ == '__main__':

    filename = raw_input('Enter dataset name MNIST/CIFAR10/LSUN: ')
    model_name = raw_input('Enter your model name (if none, leave blank): ')
    #######MNIST#########
    if (filename == 'MNIST'):
        dataset = mnist_datapath+'/mnist.pkl.gz'
        f       = gzip.open(dataset, 'rb') 
        train_set, valid_set, test_set = cPickle.load(f)   
        f.close()

        N ,D = train_set[0].shape; Nv,D = valid_set[0].shape; Nt,D = test_set[0].shape
        train_set = shared_dataset(train_set)
        valid_set = shared_dataset(valid_set)
        test_set  = shared_dataset(test_set )
     

    #######CIFAR10#######
    elif (filename == 'CIFAR10'):
        train_set, valid_set, test_set = load_cifar10(path=cifar10_datapath)
        train_set[0] = train_set[0] / 255.
        valid_set[0] = valid_set[0] / 255.
        test_set[0]  = test_set[0]  / 255.

        # print("before shared train_set[0]: ", train_set[0].shape);
        N ,D = train_set[0].shape; Nv,D = valid_set[0].shape; Nt,D = test_set[0].shape
        train_set = shared_dataset(train_set)
        valid_set = shared_dataset(valid_set)
        test_set  = shared_dataset(test_set )
     
    # print 'num_z %d' % (num_z)
    #######LSUN#######
    elif (filename == 'LSUN'):
        # store the filenames into a list.
        train_filenames = sorted(glob.glob(lsun_datapath + 'train_hkl_b100_b_100/*' + '.hkl'))
        valid_filenames = sorted(glob.glob(lsun_datapath + 'val_hkl_b100_b_100/*' + '.hkl'))
        test_filenames = sorted(glob.glob(lsun_datapath + 'test_hkl_b100_b_100/*' + '.hkl'))

        train_data = hkl.load(train_filenames[0]) / 255.
        train_data = train_data.astype('float32').transpose([3,0,1,2]);
        a,b,c,d = train_data.shape
        train_data = train_data.reshape(a,b*c*d)
        train_set = [train_data, np.zeros((a,))]
        # print (train_filenames)
        
        train_data_cllct = train_data; 
        # for purposes of setting up model.
        for i in xrange(1,len(train_filenames)):
        # for i in xrange(1,2):#TODO: find if its above forloop.
            train_data = hkl.load(train_filenames[i]) / 255.
            train_data = train_data.astype('float32').transpose([3,0,1,2]);
            a,b,c,d = train_data.shape
            train_data = train_data.reshape(a,b*c*d)
            train_data_cllct = np.vstack((train_data_cllct, train_data))
        
        # print(train_data_cllct.shape); 
        train_set = [train_data_cllct, np.zeros((a,))]
      
        val_data = hkl.load(valid_filenames[0]) / 255.
        val_data = val_data.astype('float32').transpose([3,0,1,2]);
        a,b,c,d = val_data.shape
        val_data = val_data.reshape(a, b*c*d)
        valid_set = [val_data, np.zeros((a,))]
        test_set = valid_set
        
        N ,D = train_set[0].shape; Nv,D = valid_set[0].shape; Nt,D = test_set[0].shape
        train_set = shared_dataset(train_set)
        valid_set = shared_dataset(valid_set)
        test_set  = shared_dataset(test_set)

    opt_params   = [batch_sz, epsilon, momentum, num_epoch, N, Nv, Nt,lam]    
    book_keeping = main(train_set, valid_set, test_set, opt_params,filename)


