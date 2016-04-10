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

## How to set-up LSUN dataset
1. Obtain the LSUN dataset from [fyu's repository](https://github.com/fyu/lsun)
2. Resize the image to 64x64 or 128x128.
3. Split the dataset to train/val/test set.
4. Update the paths in provided paths.yaml, and run the script 
```
python to_hkl.py <toy/full>
```
Link it to the inquire/main file, e.g.
```
lsun_datapath='/local/scratch/chris/church/preprocessed_toy_10/'
``` 

## How to run
Entry code for CIFAR10 and LSUN Church are 
```
    - ./main_granI_cifar10.py
```
## How to obtain samples with pretrained models
First download the pretrained model from this [Dropbox Link](https://www.dropbox.com/sh/1jek1alxyjhcnjh/AADOWgtWOWF-LYEuMekxe2yWa?dl=0), save it to a local folder, and supply the path when prompted.
```
    python inquire_samples.py # to attain Nearest Neighbour and Sequential Samples

    python main_granI_lsun.py # to attain 100 samples from the pretrained model.

```

Here are some CIFAR10 samples generated from GRAN:

![Image of cifar10](https://github.com/jiwoongim/GRAN/blob/master/figs/cifar10/cifar10_granI_samples.jpeg)

![Image of cifar10](https://github.com/jiwoongim/GRAN/blob/master/figs/cifar10/cifar10_granI_samples2.jpeg)

Here are some LSUN Church samples generated from GRAN:

![Image of lsun](https://github.com/jiwoongim/GRAN/blob/master/figs/lsun/lsun_ts3.jpg)

![Image of lsun](https://github.com/jiwoongim/GRAN/blob/master/figs/lsun/lsun_ts5.jpg)


