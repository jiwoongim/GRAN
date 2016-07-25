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


import sys, os
import numpy as np
import theano
import theano.tensor as T

from utils import *
import theano.sandbox.rng_mrg as RNG_MRG
rng = np.random.RandomState(1234)
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))


class Optimize():

    def __init__(self, opt_params):

        self.batch_sz, self.epsilon_gen, self.epsilon_dis, self.momentum, self.num_epoch, self.N, self.Nv, self.Nt\
                                                                    = opt_params      


    def ADAM(self, params, gparams, lr, beta1 = 0.1,beta2 = 0.001,epsilon = 1e-8, l = 1e-8):

        '''ADAM Code from 
            https://github.com/danfischetti/deep-recurrent-attentive-writer/blob/master/DRAW/adam.py
        '''
        self.m = [theano.shared(name = 'm', \
                value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in params]
        self.v = [theano.shared(name = 'v', \
        	value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in params]
        self.t = theano.shared(name = 't',value = np.asarray(1).astype(theano.config.floatX))
        updates = [(self.t,self.t+1)] 

        for param, gparam,m,v in zip(params, gparams, self.m, self.v):

            b1_t = 1-(1-beta1)*(l**(self.t-1)) 
            m_t = b1_t*gparam + (1-b1_t)*m
            updates.append((m,m_t))
            v_t = beta2*(gparam**2)+(1-beta2)*v
            updates.append((v,v_t))
            m_t_bias = m_t/(1-(1-beta1)**self.t)	
            v_t_bias = v_t/(1-(1-beta2)**self.t)
            updates.append((param,param - lr*m_t_bias/(T.sqrt(v_t_bias)+epsilon)))		

        return updates


    def MGD(self, params, gparams, lr):

        #Update momentum
        for param in model.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            deltaWs[param] = theano.shared(init)

        for param in model.params:
            updates_mom.append((param, param + deltaWs[param] * \
                            T.cast(mom, dtype=theano.config.floatX)))

        for param, gparam in zip(model.params, gparams):

            deltaV = T.cast(mom, dtype=theano.config.floatX)\
                    * deltaWs[param] - gparam * T.cast(lr, dtype=theano.config.floatX)     #new momentum

            update_grads.append((deltaWs[param], deltaV))
            new_param = param + deltaV

            update_grads.append((param, new_param))

        return update_grads


    def inspect_inputs(i, node, fn):
        print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],


    def inspect_outputs(i, node, fn):
        print "output(s) value(s):", [output[0] for output in fn.outputs]



    def optimize_gan_hkl(self, model, lam1=0.00001):
        """
        optimizer for hkl packaged dataset. 
        Returns the updates for discirminator & generator and computed costs for the model.
        """

        i = T.iscalar('i'); 
        lr = T.fscalar('lr');
        Xu = T.fmatrix('X'); 

        cost_disc   = model.cost_dis(Xu, self.batch_sz) \
                                + lam1 * model.dis_network.weight_decay_l2()
        gparams_dis = T.grad(cost_disc, model.dis_network.params)

        cost_gen    = model.cost_gen(self.batch_sz) 
        gparams_gen = T.grad(cost_gen, model.gen_network.params)


        updates_dis = self.ADAM(model.dis_network.params, gparams_dis, lr)
        updates_gen = self.ADAM(model.gen_network.params, gparams_gen, lr)
        
        
        discriminator_update = theano.function([Xu, theano.Param(lr,default=self.epsilon_dis)],\
                                    outputs=cost_disc, updates=updates_dis)

        generator_update = theano.function([theano.Param(lr,default=self.epsilon_gen)],\
                outputs=cost_gen, updates=updates_gen)

        get_valid_cost   = theano.function([Xu], outputs=[cost_disc, cost_gen])

        get_test_cost   = theano.function([Xu], outputs=[cost_disc, cost_gen])

        return discriminator_update, generator_update, get_valid_cost, get_test_cost


    def optimize_gan(self, model, train_set, valid_set, test_set, lam1=0.00001):
        """
        optimizer for non packaged dataset, 
        returning updates for discriminator & generator, as well as the computed costs.
        """

        i = T.iscalar('i'); lr = T.fscalar('lr');
        Xu = T.matrix('X'); 
        cost_disc   = model.cost_dis(Xu, self.batch_sz) \
                     + lam1 * model.dis_network.weight_decay_l2() 

        gparams_dis = T.grad(cost_disc, model.dis_network.params)

        cost_gen    = model.cost_gen(self.batch_sz)
        gparams_gen = T.grad(cost_gen, model.gen_network.params)


        updates_dis = self.ADAM(model.dis_network.params, gparams_dis, lr)
        updates_gen = self.ADAM(model.gen_network.params, gparams_gen, lr)

        discriminator_update = theano.function([i, theano.Param(lr,default=self.epsilon_dis)],\
                outputs=cost_disc, updates=updates_dis,\
                givens={Xu:train_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        generator_update = theano.function([theano.Param(lr,default=self.epsilon_gen)],\
                outputs=cost_gen, updates=updates_gen)

        get_valid_cost   = theano.function([i], outputs=[cost_disc, cost_gen],\
                givens={Xu:valid_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        get_test_cost   = theano.function([i], outputs=[cost_disc, cost_gen],\
                givens={Xu:test_set[0][i*self.batch_sz:(i+1)*self.batch_sz]})

        return discriminator_update, generator_update, get_valid_cost, get_test_cost
    
    
    def get_samples(self, model):

        num_sam = T.iscalar('i'); 
        return theano.function([num_sam], model.get_samples(num_sam, num_steps=9))


    def get_seq_drawing(self, model):
        
        num_sam = T.iscalar('i'); 
        return theano.function([num_sam], model.sequential_drawing(num_sam))

