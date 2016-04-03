# Generating Images with Recurrent Adversarial Networks 

Python (Theano) implementation of Generating Images with Recurrent Adversarial Networks code provided 
by Daniel Jiwoong Im, Chris Dongjoo Kim, Hui Jiang, and Roland, Memisevic 

Generative Recurrent Adversarial Network (GRAN) is a recurrent generative model inspired by
the view that unrolling the gradient-based optimization yields
a recurrent computation that creates images by
incrementally adding onto a visual “canvas”.
GRAN is trained using adversarial training to generate very good image
samples. 

Generative Adversarial Metric (GAM) quantitatively
compare adversarial networks by having
the generators and discriminators of these networks
compete against each other.

For more information, see 
```bibtex
@article{Im2015,
    title={Generating Images with Recurrent Adversarial Networks },
    author={Im, Daniel Jiwoong and Kim, Chris Dongjoo and Jiang, Hui and Memisevic, Roland},
    journal={http://arxiv.org/abs/1602.05110},
    year={2016}
}
```
If you use this in your research, we kindly ask that you cite the above arxiv paper.


## Dependencies
Packages
* [numpy](http://www.numpy.org/)
* [Theano ('0.7.0.dev-725b7a3f34dd582f9aa2071a5b6caedb3091e782')](http://deeplearning.net/software/theano/) 


## How to run
Entry code for CIFAR10 and LSUN Church are 
```
    - ./main_granI_cifar10.py
```

Here are some CIFAR10 samples generated from GRAN:
![Image of cifar10](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/cifar10_granI_samples.pdf)
![Image of cifar10](https://raw.githubusercontent.com/jiwoongim/DVAE/master/figs/cifar10_granI_samples2.pdf)

