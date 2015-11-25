
Bidirectional Helmholtz Machines
================================

This repository contains the source code and additional results for the 
experiments described in 

 http://arxiv.org/abs/1506.03877

Overview
--------

![concept](/doc/bihm-concept.png)

The basic idea is to create a deep generative model for unsupervised learning
by combining a top-down directed model P and a bottom up directed model Q into
a joint model P\*. We show that we can train P\* such that P and Q are useful
approximate inference distributions when we want to sample from the model, or
when we want to perform inference.

We generally observe that BiHMs prefer deep architectures with many layers of
latent variables. I.e., our best model for the binarized MNIST dataset has 12
layers with 300,200,100,75,50,35,30,25,20,15,10,10 binary latent units. This
model reaches a test set LL of 84.8 nats.

 
### Samples from the model ###

![bmnist-samples](/doc/bmnist-bihm-samples000.png)
![bmnist-samples](/doc/bmnist-bihm-samples.gif)

The left image shows 100 random samples from the top-down model P; the right
image shows that starting from this point and running 250 Gibbs MCMC steps to
approximately sample from P\* results in higher quality, crisp digits. 
(we visualize the Bernoulli probability per pixel instead of sampling from it)

### Inpainting ###

![bmnist-inpainting](/doc/bmnist-bihm-inpaint000.png)
![bmnist-inpainting](/doc/bmnist-bihm-inpaint-sampled.gif)

The left image shows 10 different digits that have been partially occluded. For each digit, 
we sample 10 different starting configurations from Q and subsequently run a Markov chain 
that produces approx. samples from P\* which are consistent with the initial digits.


Dependencies
------------

This code depends on [Fuel](https://github.com/mila-udem/fuel), 
[Theano](https://github.com/Theano/Theano), [Blocks](https://github.com/mila-udem/blocks) 
and various other libraries from the scientific python universe.

