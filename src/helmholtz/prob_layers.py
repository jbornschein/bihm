
from __future__ import division, print_function 

import logging

import numpy
import theano

from collections import OrderedDict
from theano import tensor

from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.roles import add_role, PARAMETER, WEIGHT, BIAS
from blocks.bricks import Random, MLP, Initializable
from blocks.utils import pack, shared_floatx_zeros
from blocks.select import Selector

from .distributions import bernoulli

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

N_STREAMS = 2048

sigmoid_frindge = 1e-6
 
#-----------------------------------------------------------------------------

class ProbabilisticTopLayer(Random):
    def __init__(self, **kwargs):
        super(ProbabilisticTopLayer, self).__init__(**kwargs)

    def sample_expected(self):
        raise NotImplemented

    def sample(self):
        raise NotImplemented

    def log_prob(self, X):
        raise NotImplemented

    def get_gradients(self, X, weights=1.):
        cost = -(weights * self.log_prob(X)).sum()
 
        params = Selector(self).get_parameters()
        
        gradients = OrderedDict()
        if isinstance(weights, float):
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(cost, param, consider_constant=[X])
        else:
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(cost, param, consider_constant=[X, weights])
            
        return gradients


class ProbabilisticLayer(Random):
    def __init__(self, **kwargs):
        super(ProbabilisticLayer, self).__init__(**kwargs)

    def sample_expected(self, Y):
        raise NotImplemented

    def sample(self, Y):
        raise NotImplemented

    def log_prob(self, X, Y):
        raise NotImplemented

    def get_gradients(self, X, Y, weights=1.):
        cost = -(weights * self.log_prob(X, Y)).sum()
        
        params = Selector(self).get_parameters()

        gradients = OrderedDict()
        if isinstance(weights, float):
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(cost, param, consider_constant=[X, Y])
        else:
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(cost, param, consider_constant=[X, Y, weights])
            
        return gradients

#-----------------------------------------------------------------------------

class BernoulliTopLayer(Initializable, ProbabilisticTopLayer):
    def __init__(self, dim_X, biases_init, **kwargs):
        super(BernoulliTopLayer, self).__init__(**kwargs)
        self.dim_X = dim_X
        self.biases_init = biases_init

    def _allocate(self):
        b = shared_floatx_zeros((self.dim_X,), name='b')
        add_role(b, BIAS)
        self.parameters.append(b)
        self.add_auxiliary_variable(b.norm(2), name='b_norm')
        
    def _initialize(self):
        b, = self.parameters
        self.biases_init.initialize(b, self.rng)

    @application(inputs=[], outputs=['X_expected'])
    def sample_expected(self):
        b = self.parameters[0]
        return tensor.nnet.sigmoid(b).clip(sigmoid_frindge, 1.-sigmoid_frindge)

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        prob_X = self.sample_expected()
        prob_X = tensor.zeros((n_samples, prob_X.shape[0])) + prob_X
        X = bernoulli(prob_X, rng=self.theano_rng, nstreams=N_STREAMS)
        return X, self.log_prob(X)

    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        prob_X = self.sample_expected()
        log_prob = X*tensor.log(prob_X) + (1.-X)*tensor.log(1.-prob_X)
        return log_prob.sum(axis=1)


class BernoulliLayer(Initializable, ProbabilisticLayer):
    def __init__(self, mlp, **kwargs):
        super(BernoulliLayer, self).__init__(**kwargs)

        self.mlp = mlp

        self.dim_X = mlp.output_dim
        self.dim_Y = mlp.input_dim

        self.children = [self.mlp]

    @application(inputs=['Y'], outputs=['X_expected'])
    def sample_expected(self, Y):
        return self.mlp.apply(Y).clip(sigmoid_frindge, 1.-sigmoid_frindge)

    @application(inputs=['Y'], outputs=['X', 'log_prob'])
    def sample(self, Y):
        prob_X = self.sample_expected(Y)
        X = bernoulli(prob_X, rng=self.theano_rng, nstreams=N_STREAMS)
        return X, self.log_prob(X, Y)

    @application(inputs=['X', 'Y'], outputs=['log_prob'])
    def log_prob(self, X, Y):
        prob_X = self.sample_expected(Y)
        log_prob = X*tensor.log(prob_X) + (1.-X)*tensor.log(1-prob_X)
        return log_prob.sum(axis=1)

#-----------------------------------------------------------------------------


class GaussianTopLayer(Initializable, ProbabilisticTopLayer):
    def __init__(self, dim_X, fixed_sigma=None, **kwargs):
        super(GaussianTopLayer, self).__init__(**kwargs)
        self.fixed_sigma = fixed_sigma
        self.dim_X = dim_X

    def _allocate(self):
        b = shared_floatx_zeros((self.dim_X,), name='b')
        add_role(b, BIAS)
        self.parameters = [b]
        
    def _initialize(self):
        b, = self.parameters
        self.biases_init.initialize(b, self.rng)

    @application(inputs=[], outputs=['mean', 'log_sigma'])
    def sample_expected(self, n_samples):
        b, = self.parameters
        mean      = tensor.zeros((n_samples, self.dim_X))
        #log_sigma = tensor.zeros((n_samples, self.dim_X)) + b
        log_sigma = tensor.log(self.fixed_sigma)

        return mean, log_sigma

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        mean, log_sigma = self.sample_expected(n_samples)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=(n_samples, self.dim_X),
                    avg=0., std=1.)
        # ... and scale/translate samples
        X = mean + tensor.exp(log_sigma) * U

        return X, self.log_prob(X)

    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        mean, log_sigma = self.sample_expected(X.shape[0])

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*tensor.log(2*numpy.pi) - log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)

        return log_prob.sum(axis=1)


#-----------------------------------------------------------------------------


class GaussianLayerFixedSigma(Initializable, ProbabilisticLayer):
    def __init__(self, dim_X, mlp, sigma=None, **kwargs):
        super(GaussianLayerFixedSigma, self).__init__(**kwargs)
        self.mlp = mlp
        self.dim_X = dim_X
        self.dim_Y = mlp.input_dim
        self.dim_H = mlp.output_dim
        self.sigma = sigma

        self.children = [self.mlp]
        

    def _allocate(self):
        super(GaussianLayerFixedSigma, self)._allocate()

        dim_X, dim_H = self.dim_X, self.dim_H

        self.W_mean = shared_floatx_zeros((dim_H, dim_X), name='W_mean')
        add_role(self.W_mean, WEIGHT)

        self.b_mean = shared_floatx_zeros((dim_X,), name='b_mean')
        add_role(self.b_mean, BIAS)

        self.parameters = [self.W_mean, self.b_mean]
        
    def _initialize(self):
        super(GaussianLayerFixedSigma, self)._initialize()

        W_mean, b_mean = self.parameters

        self.weights_init.initialize(W_mean, self.rng)
        self.biases_init.initialize(b_mean, self.rng)

    @application(inputs=['Y'], outputs=['mean', 'log_sigma'])
    def sample_expected(self, Y):
        W_mean, b_mean = self.parameters

        a = self.mlp.apply(Y)
        mean      = tensor.dot(a, W_mean) + b_mean
        log_sigma = tensor.log(self.sigma)

        return mean, log_sigma

    @application(inputs=['Y'], outputs=['X', 'log_prob'])
    def sample(self, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=mean.shape, 
                    avg=0., std=1.)
        # ... and scale/translate samples
        X = mean + tensor.exp(log_sigma) * U

        return X, self.log_prob(X, Y)

    @application(inputs=['X', 'Y'], outputs=['log_prob'])
    def log_prob(self, X, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*tensor.log(2*numpy.pi) - log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)

        return log_prob.sum(axis=1)


#-----------------------------------------------------------------------------


class GaussianLayer(Initializable, ProbabilisticLayer):
    def __init__(self, dim_X, mlp, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.mlp = mlp
        self.dim_X = dim_X
        self.dim_Y = mlp.input_dim
        self.dim_H = mlp.output_dim

        self.children = [self.mlp]

    def _allocate(self):
        super(GaussianLayer, self)._allocate()

        dim_X, dim_Y, dim_H = self.dim_X, self.dim_Y, self.dim_H

        self.W_mean = shared_floatx_zeros((dim_H, dim_X), name='W_mean')
        self.W_ls   = shared_floatx_zeros((dim_H, dim_X), name='W_ls')
        add_role(self.W_mean, WEIGHT)
        add_role(self.W_ls, WEIGHT)

        self.b_mean = shared_floatx_zeros((dim_X,), name='b_mean')
        self.b_ls   = shared_floatx_zeros((dim_X,), name='b_ls')
        add_role(self.b_mean, BIAS)
        add_role(self.b_ls, BIAS)

        self.parameters = [self.W_mean, self.W_ls, self.b_mean, self.b_ls]
        
    def _initialize(self):
        super(GaussianLayer, self)._initialize()

        W_mean, W_ls, b_mean, b_ls = self.parameters

        self.weights_init.initialize(W_mean, self.rng)
        self.weights_init.initialize(W_ls, self.rng)
        self.biases_init.initialize(b_mean, self.rng)
        self.biases_init.initialize(b_ls, self.rng)

    @application(inputs=['Y'], outputs=['mean', 'log_sigma'])
    def sample_expected(self, Y):
        W_mean, W_ls, b_mean, b_ls = self.parameters

        a = self.mlp.apply(Y)
        mean = tensor.dot(a, W_mean) + b_mean
        log_sigma = tensor.dot(a, W_ls) + b_ls

        return mean, log_sigma

    @application(inputs=['Y'], outputs=['X', 'log_prob'])
    def sample(self, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=mean.shape, 
                    avg=0., std=1.)
        # ... and scale/translate samples
        X = mean + tensor.exp(log_sigma) * U

        return X, self.log_prob(X, Y)

    @application(inputs=['X', 'Y'], outputs=['log_prob'])
    def log_prob(self, X, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*tensor.log(2*numpy.pi) - log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)

        return log_prob.sum(axis=1)

    def get_gradients(self, X, Y, weights=1.):
        W_mean, W_ls, b_mean, b_ls = self.parameters

        mean, log_sigma = self.sample_expected(Y)
        sigma = tensor.exp(log_sigma)

        cost = -log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)
        if weights != 1.:
            cost = -weights.dimshuffle(0, 'x') * cost

        cost_scaled = sigma**2 * cost
        cost_gscale = (sigma**2).sum(axis=1).dimshuffle([0, 'x'])   
        cost_gscale = cost_gscale * cost
        
        gradients = OrderedDict()

        params = Selector(self.mlp).get_parameters()
        for pname, param in params.iteritems():
            gradients[param] = tensor.grad(cost_gscale.sum(), param, consider_constant=[X, Y])

        gradients[W_mean] = tensor.grad(cost_scaled.sum(), W_mean, consider_constant=[X, Y])
        gradients[b_mean] = tensor.grad(cost_scaled.sum(), b_mean, consider_constant=[X, Y])

        gradients[W_ls] = tensor.grad(cost_scaled.sum(), W_ls, consider_constant=[X, Y])
        gradients[b_ls] = tensor.grad(cost_scaled.sum(), b_ls, consider_constant=[X, Y])
            
        return gradients
