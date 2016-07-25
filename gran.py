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

import theano 
import numpy as np
import scipy as sp
from convnet_cuda32 import *
from convnet_cuda28 import *
from convnet_cuda64 import *
from convnet_cuda128 import *

from recGenI32 import *
from recGenI28 import *
from recGenI64 import *
from recGenI128 import *

class GRAN():

    def __init__(self, model_params):
        """Initializes discriminator and generator for 64x64/128x128/MNIST"""
    
        gen_params, disc_params = model_params
        self.num_steps          = gen_params[-1]

        if gen_params[1] == 784: #MNIST
            self.dis_network = convnet28(disc_params) 
            self.gen_network = RecGenI28(gen_params)
        elif gen_params[1] == 64*64*3:
            self.dis_network = convnet64(disc_params) 
            self.gen_network = RecGenI64(gen_params)
        elif gen_params[1] == 128*128*3:
            self.dis_network = convnet128(disc_params) 
            self.gen_network = RecGenI128(gen_params)
        else:   
            ##32x32x3, i.e., CIFAR10 would fall here
            self.dis_network = convnet32(disc_params) 
            self.gen_network = RecGenI32(gen_params)

        params      = self.dis_network.params + self.gen_network.params
        self.params = OrderedDict()
        for param in params:
            self.params[param.name] = param


    def cost_dis(self, X, num_examples):
        """compute cost of the discriminator"""

        target1  = T.alloc(1., num_examples)
        p_y__x1  = self.dis_network.propagate(X).flatten()

        target0      = T.alloc(0., num_examples)
        gen_samples  = self.gen_network.get_samples(num_examples)[0]
        p_y__x0      = self.dis_network.propagate(gen_samples).flatten()

        return T.mean(T.nnet.binary_crossentropy(p_y__x1, target1)) \
                        + T.mean(T.nnet.binary_crossentropy(p_y__x0, target0))

    def cost_gen(self, num_examples):
        """compute cost of the generator"""

        target      = T.alloc(1., num_examples)
        gen_samples = self.gen_network.get_samples(num_examples)[0]
        p_y__x      = self.dis_network.propagate(gen_samples).flatten()

        return T.mean(T.nnet.binary_crossentropy(p_y__x, target))

    
    def sequential_drawing(self, num_examples):
        """Fetches the sequential output of GRAN at each timestep"""
        
        canvas = self.gen_network.get_samples(num_examples)[1]
        sequential_sams = []
        for i in xrange(self.num_steps):
            sequential_sams.append(T.nnet.sigmoid(T.sum(T.stacklists(canvas[:i+1]), axis=0)))
       
        return T.stacklists(sequential_sams)

 
    def get_samples(self, num_sam, num_steps=None):
        """Fetches the generated samples"""

        if num_steps is None : num_steps = self.num_steps

        #Returns tensor (num_sam, 3, D, D)
        return self.gen_network.get_samples(num_sam, num_steps=num_steps)[0]



