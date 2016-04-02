''' Version 1.000
 Code provided by Daniel Jiwoong Im 
 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

import os, sys, math
import numpy as np
import pylab as pl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#cifar10_path='/data/lisa/data/cifar10/cifar-10-batches-py/'
#cifar10_path = '/export/mlrg/imj/machine_learning/data/cifar10/cifar-10-batches-py/'
cifar10_path='/eecs/research/asr/chris/DG_project/dataset/cifar-10-batches-py'

from utils import *
from PIL import Image


def load_cifar10(path=cifar10_path):
    '''processes the raw downloaded cifar10 dataset, and returns test/val/train set'''

    data_batch1 = unpickle(path+'data_batch_1')
    data_batch2 = unpickle(path+'data_batch_2')
    data_batch3 = unpickle(path+'data_batch_3')
    data_batch4 = unpickle(path+'data_batch_4')
    data_batch5 = unpickle(path+'data_batch_5')
    test_batch  = unpickle(path+'test_batch')

    data_batch = {}
    data_batch['data'] = np.concatenate((data_batch1['data'], data_batch2['data']), axis=0)
    data_batch['data'] = np.concatenate((data_batch['data'],  data_batch3['data']), axis=0)
    data_batch['data'] = np.concatenate((data_batch['data'],  data_batch4['data']), axis=0)
    data_batch['data'] = np.concatenate((data_batch['data'],  data_batch5['data']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch1['labels'], data_batch2['labels']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch['labels'], data_batch3['labels']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch['labels'], data_batch4['labels']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch['labels'], data_batch5['labels']), axis=0)
    test_set = [test_batch['data'], np.asarray(test_batch['labels'], dtype='float32')]

    data = gen_train_valid_test(data_batch['data'],data_batch['labels'],8,1,1) 
    train_set, valid_set, _ = data[0], data[1], data[2]
   
    return train_set, valid_set, test_set 


def load_cifar2(path=cifar10_path):

    data_batch1 = unpickle(path+'data_batch_1')
    data_batch2 = unpickle(path+'data_batch_2')
    data_batch3 = unpickle(path+'data_batch_3')
    data_batch4 = unpickle(path+'data_batch_4')

    data_batch = {}
    data_batch['data'] = np.concatenate((data_batch1['data'], data_batch2['data']), axis=0)
    data_batch['data'] = np.concatenate((data_batch['data'],  data_batch3['data']), axis=0)
    data_batch['data'] = np.concatenate((data_batch['data'],  data_batch4['data']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch1['labels'], data_batch2['labels']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch['labels'], data_batch3['labels']), axis=0)
    data_batch['labels'] = np.concatenate((data_batch['labels'], data_batch4['labels']), axis=0)


    data = gen_train_valid_test(data_batch['data'],data_batch['labels'],7,1,2) 
    train_set, valid_set, test_set = data[0], data[1], data[2] 
    
    return train_set, valid_set, test_set 

def display_images(images, tile_shape=(10,10), img_shape=(32,32), fname=None):
    """
    Displays mxn images from the dataset
    """

    DD = img_shape[0] * img_shape[1]
    images = (images[:, 0:DD],images[:, DD:DD*2],images[:, DD*2:DD*3], None)
    x = tile_raster_images(images, img_shape=img_shape, \
     						tile_shape=tile_shape, tile_spacing=(1,1), output_pixel_vals=False, scale_rows_to_unit_interval=False)


    image = Image.fromarray(np.uint8(x[:,:,0:3]))
    if fname:
        image.save(fname +'.pdf')


def display_cifar10(data_batch):
    '''get samples for cifar10 original'''
    N = data_batch['data'].shape[1]
    input = data_batch['data'][0:100,:]
    display_images(input, tile_shape=(10,10), fname='./cifar10.png')


def display_cifar10_grey(data_batch):
    '''return grey scale samples for cifar10'''
    N = data_batch['data'].shape[1]
    input = data_batch['data'][0:100,:]
    input = (input[:, 0:1024],input[:, 1024:2048],input[:, 2048:3072], None)

    x = tile_raster_images(input, img_shape=(32,32), \
     						tile_shape=(10,10), tile_spacing=(1,1), output_pixel_vals=False, scale_rows_to_unit_interval=False)

    plt.figure(1)
    plt.imshow(x,cmap='gray')


def get_cifar2(data, class1=2, class2=3):
    data1   = data['data'  ][data['labels'] == class1]
    data2   = data['data'  ][data['labels'] == class2]
    labels1 = np.zeros((len(data['labels'][data['labels'] == class1]),))
    labels2 = np.ones ((len(data['labels'][data['labels'] == class2]),))

    data  = np.concatenate((data1,data2),0)
    label = np.concatenate((labels1, labels2), 0)
    return gen_train_valid_test(data, label, 7,1,2)



