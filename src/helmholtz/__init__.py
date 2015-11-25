
from __future__ import division, print_function 

import sys
sys.path.append("../")


import logging
import numpy
import re
import theano

from abc import ABCMeta, abstractmethod
from theano import tensor
from collections import OrderedDict

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks import Random, Initializable, MLP, Tanh, Logistic
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity

from initialization import RWSInitialization
from prob_layers import BernoulliTopLayer, BernoulliLayer

logger = logging.getLogger(__name__)
floatX = theano.config.floatX


def logsumexp(A, axis=None):
    """Numerically stable log( sum( exp(A) ) ) """
    A_max = tensor.max(A, axis=axis, keepdims=True)
    B = tensor.log(tensor.sum(tensor.exp(A-A_max), axis=axis, keepdims=True))+A_max
    B = tensor.sum(B, axis=axis)
    return B


def replicate_batch(A, repeat):
    """Extend the given 2d Tensor by repeating reach line *repeat* times.

    With A.shape == (rows, cols), this function will return an array with
    shape (rows*repeat, cols).

    Parameters
    ----------
    A : T.tensor
        Each row of this 2d-Tensor will be replicated *repeat* times
    repeat : int

    Returns
    -------
    B : T.tensor
    """
    A_ = A.dimshuffle((0, 'x', 1))
    A_ = A_ + tensor.zeros((A.shape[0], repeat, A.shape[1]), dtype=floatX)
    A_ = A_.reshape( [A_.shape[0]*repeat, A.shape[1]] )
    return A_


def flatten_values(vals, size):
    """ Flatten a list of Theano tensors.
    
    Flatten each Theano tensor in *vals* such that each of them is 
    reshaped from shape (a, b, *c) to (size, *c). In other words:
    The first two dimension of each tensor in *vals* are replaced 
    with a single dimension is size *size*.

    Parameters
    ----------
    vals : list
        List of Theano tensors

    size : int
        New size of the first dimension 
    
    Returns
    -------
    flattened_vals : list
        Reshaped version of each *vals* tensor.
    """
    data_dim = vals[0].ndim - 2
    assert all([v.ndim == data_dim+2 for v in vals])

    if data_dim == 0:
        return [v.reshape([size]) for v in vals]
    elif data_dim == 1:
        return [v.reshape([size, v.shape[2]]) for v in vals]
    raise 

def unflatten_values(vals, batch_size, n_samples):
    """ Reshape a list of Theano tensors. 

    Parameters
    ----------
    vals : list
        List of Theano tensors
    batch_size : int
        New first dimension 
    n_samples : int
        New second dimension
    
    Returns
    -------
    reshaped_vals : list
        Reshaped version of each *vals* tensor.
    """
    data_dim = vals[0].ndim - 1
    assert all([v.ndim == data_dim+1 for v in vals])

    if data_dim == 0:
        return [v.reshape([batch_size, n_samples]) for v in vals]
    elif data_dim == 1:
        return [v.reshape([batch_size, n_samples, v.shape[1]]) for v in vals]
    raise 


def merge_gradients(old_gradients, new_gradients, scale=1.):
    """Take and merge multiple ordered dicts 
    """
    if isinstance(new_gradients, (dict, OrderedDict)):
        new_gradients = [new_gradients]

    for gradients in new_gradients:
        assert isinstance(gradients, (dict, OrderedDict))
        for key, val in gradients.items():
            if old_gradients.has_key(key):
                old_gradients[key] = old_gradients[key] + scale * val
            else:       
                old_gradients[key] = scale * val
    return old_gradients


def create_layers(layer_spec, data_dim, deterministic_layers=0, deterministic_act=None, deterministic_size=1.):
    """
    Parameters
    ----------
    layer_spec : str
        A specification for the layers to construct; typically takes a string
        like "100,50,25,10" and create P- and Q-models with  4 hidden layers
        of specified size.
    data_dim : int
        Dimensionality of the trainig/test data. The bottom-most layers
        will work with thgis dimension.
    deterministic_layers : int
        Dont want to talk about it.
    deterministic_act : 
    deterministic_size : float

    Returns
    -------
    p_layers : list
        List of ProbabilisticLayers with a ProbabilisticTopLayer on top.
    q_layers : list
        List of ProbabilisticLayers
    """
    inits = {
        'weights_init': RWSInitialization(factor=1.),
#        'weights_init': IsotropicGaussian(0.1),
        'biases_init': Constant(-1.0),
    }

    m = re.match("(\d*\.?\d*)x-(\d+)l-(\d+)", layer_spec)
    if m:
        first = int(data_dim * float(m.groups()[0]))
        last = float(m.groups()[2])
        n_layers = int(m.groups()[1])

        base = numpy.exp(numpy.log(first/last) / (n_layers-1))
        layer_sizes = [data_dim] + [int(last*base**i) for i in reversed(range(n_layers))]
        print(layer_sizes)
    else:
        layer_specs = [i for i in layer_spec.split(",")]
        layer_sizes = [data_dim] + [int(i) for i in layer_specs]

    p_layers = []
    q_layers = []
    for l, (size_lower, size_upper) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        """
        if size_upper < 0:
            lower_before_repeat = size_lower
            p = BernoulliLayer(
                    MLP([Sigmoid()], [size_lower, size_lower], **rinits), 
                    name="p_layer%d"%l)
            q = BernoulliLayer(
                    MLP([Sigmoid()], [size_lower, size_lower], **rinits), 
                    name="q_layer%d"%l)
            for r in xrange(-size_upper):
                p_layers.append(p)
                q_layers.append(q)
            continue
        elif size_lower < 0:
            size_lower = lower_before_repeat
        """
        size_mid = (deterministic_size * (size_upper + size_lower)) // 2

        p_layers.append(
            BernoulliLayer(
                MLP(
                    [deterministic_act() for i in range(deterministic_layers)]+[Logistic()],
                    [size_upper]+[size_mid for i in range(deterministic_layers)]+[size_lower],
                    **inits), 
                name="p_layer%d"%l))
        q_layers.append(
            BernoulliLayer(
                MLP(
                    [deterministic_act() for i in range(deterministic_layers)]+[Logistic()],
                    [size_lower]+[size_mid for i in range(deterministic_layers)]+[size_upper],
                    **inits), 
                name="q_layer%d"%l))

    p_layers.append(
        BernoulliTopLayer(
            layer_sizes[-1],
            name="p_top_layer",
            **inits))

    return p_layers, q_layers

#-----------------------------------------------------------------------------


class HelmholtzMachine(Initializable, Random):
    def __init__(self, p_layers, q_layers, **kwargs):
        super(HelmholtzMachine, self).__init__(**kwargs)
        
        self.p_layers = p_layers
        self.q_layers = q_layers

        self.children = p_layers + q_layers

#-----------------------------------------------------------------------------


class GradientMonitor(object):
    def __init__(self, gradients, prefix=""):
        self.gradients = gradients
        self.prefix = prefix
        pass

    def vars(self):
        prefix = self.prefix 
        monitor_vars = []

        aggregators = {
            'min':   tensor.min,
            'max':   tensor.max,
            'mean':  tensor.mean,
        }

        for key, value in six.iteritems(self.gradients):
            min = tensor.min(value)
            ipdb.set_trace()
            monitor_vars.append(monitor_vars)

        return monitor_vars
