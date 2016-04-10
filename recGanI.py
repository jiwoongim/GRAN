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

'''This is same script as GRAN.py; Duplicate for now.. '''
import theano 
import numpy as np
import scipy as sp
from convnet_cuda32 import *
from convnet_cuda28 import *
from convnet_cuda64 import *

from batch_norm_conv_layer import *
from recGenI32 import *
from recGenI28 import *
from recGenI64 import *

class RecGanI():

    def __init__(self, model_params):
    
        gen_params, disc_params = model_params
        self.num_steps   = gen_params[-1]
        if gen_params[1] == 784:
            self.dis_network = convnet28(disc_params) 
            self.gen_network = RecGenI28(gen_params)
        elif gen_params[1] == 64*64*3:
            self.dis_network = convnet64(disc_params) 
            self.gen_network = RecGenI64(gen_params)
        else:
            # cifar10 would fall here.
            self.dis_network = convnet32(disc_params) 
            self.gen_network = RecGenI32(gen_params)
        self.params = self.dis_network.params + self.gen_network.params


    def cost_dis(self, X, num_examples):

        target1  = T.alloc(1., num_examples)
        p_y__x1  = self.dis_network.propagate(X).flatten()

        target0      = T.alloc(0., num_examples)
        gen_samples  = self.gen_network.get_samples(num_examples)[0]
        p_y__x0      = self.dis_network.propagate(gen_samples).flatten()

        return T.mean(T.nnet.binary_crossentropy(p_y__x1, target1)) \
                        + T.mean(T.nnet.binary_crossentropy(p_y__x0, target0))

    def cost_gen(self, num_examples):

        target      = T.alloc(1., num_examples)
        gen_samples = self.gen_network.get_samples(num_examples)[0]
        p_y__x      = self.dis_network.propagate(gen_samples).flatten()

        return T.mean(T.nnet.binary_crossentropy(p_y__x, target))

    
    def sequential_drawing(self, num_examples):
        
        canvas = self.gen_network.get_samples(num_examples)[1]
        sequential_sams = []
        for i in xrange(self.num_steps):
            sequential_sams.append(T.nnet.sigmoid(T.sum(T.stacklists(canvas[:i+1]), axis=0)))
       
        return T.stacklists(sequential_sams)

    
    def get_samples(self, num_sam):

        #Returns tensor (num_sam, 3, D, D)
        return self.gen_network.get_samples(num_sam)[0]



