#!/usr/bin/env python 

from __future__ import division, print_function

import logging

import theano
import theano.tensor as tensor

import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.opt import register_canonicalize, register_specialize
from theano.gradient import grad_undefined
from theano import gof

logger = logging.getLogger(__name__)

N_STREAMS = 20148
floatX = theano.config.floatX
theano_rng = MRG_RandomStreams(seed=2341)

#-----------------------------------------------------------------------------

class BernoulliOp(theano.Op):
    """ back-propable' Bernoulli distribution.
    """
    #__props__ = ('theano_rng')

    def __init__(self):
        super(BernoulliOp, self).__init__()

    def make_node(self, prob, rng=None, nstreams=None):
        assert hasattr(self, '_props')

        if rng is None:
            rng = theano_rng
        if nstreams is None:
            nstreams = N_STREAMS

        prob = theano.tensor.as_tensor_variable(prob)
        noise = rng.uniform(size=prob.shape, nstreams=nstreams)

        return theano.Apply(self, [prob, noise], [prob.type()])

    def perform(self, node, inputs, output_storage):
        logger.warning("BernoulliOp.perform(...) called")
        
        prob = inputs[0]
        noise = inputs[1]

        samples = output_storage[0]
        samples[0] = (noise < prob).astype(floatX)

    def grad(self, inputs, grads):
        logger.warning("BernoulliOp.grad(...) called")

        prob = inputs[0]
        noise = inputs[1]
        #import ipdb; ipdb.set_trace()

        #g0 = prob.zeros_like().astype(theano.config.floatX)
        g0 = prob * grads[0]
        g1 = grad_undefined(self, 1, noise)
        return [g0, g1]

bernoulli = BernoulliOp()

#-----------------------------------------------------------------------------
# Optimization

@register_canonicalize
@register_specialize
@gof.local_optimizer([BernoulliOp])
def replace_bernoulli_op(node):
    if not isinstance(node.op, BernoulliOp):
        return False

    prob = node.inputs[0]
    noise = node.inputs[1]

    
    
    samples = (noise < prob).astype(floatX)
    
    return [samples]

#=============================================================================

if __name__ == "__main__":
    n_samples = tensor.iscalar("n_samples")
    prob = tensor.vector('prob')
    target_prob = tensor.vector('target_prob')

    shape = (n_samples, prob.shape[0])
    bprob = tensor.ones(shape) * prob

    samples = bernoulli(bprob, rng=theano_rng)
    
    mean = tensor.mean(samples, axis=0)
    cost = tensor.sum((mean-target_prob)**2)
    
    grads = theano.grad(cost, prob)

    print("-"*78)
    print(theano.printing.debugprint(samples))
    print("-"*78)

    do_sample = theano.function(
                inputs=[prob, target_prob, n_samples], 
                outputs=[samples, grads],
                allow_input_downcast=True, name="do_sample")

    
    #-------------------------------------------------------------------------
    n_samples = 10000
    prob = np.linspace(0, 1, 10)
    target_prob = prob

    samples, grads = do_sample(prob, target_prob, n_samples)
    print("== samples =========")
    print(samples)
    print("== mean ============")
    print(np.mean(samples, axis=0))
    print("== grads ===========")
    print(grads)
