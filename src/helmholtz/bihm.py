
from __future__ import division, print_function 

import sys

import re
import logging

import numpy
import theano

from theano import tensor
from collections import OrderedDict

from blocks.bricks.base import application, Brick, lazy
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity
from blocks.roles import has_roles, WEIGHT
from blocks.select import Selector

from . import HelmholtzMachine
from . import merge_gradients, flatten_values, unflatten_values, replicate_batch, logsumexp

logger = logging.getLogger(__name__)
floatX = theano.config.floatX


#-----------------------------------------------------------------------------


class BiHM(HelmholtzMachine):
    def __init__(self, p_layers, q_layers, l1reg=0.0, l2reg=0.0, transpose_init=False, **kwargs):
        super(BiHM, self).__init__(p_layers, q_layers, **kwargs)

        self.transpose_init = transpose_init
        self.l1reg = l1reg
        self.l2reg = l2reg
        self.zreg = 0.0

        self.children = p_layers + q_layers

    def log_prob_p(self, samples):
        """ Calculate p(h_l | h_{l+1}) for all layers.  """
        n_layers = len(self.p_layers)
        n_samples = samples[0].shape[0]

        log_p = [None] * n_layers
        for l in xrange(n_layers-1):
            log_p[l] = self.p_layers[l].log_prob(samples[l], samples[l+1])
        log_p[n_layers-1] = self.p_layers[n_layers-1].log_prob(samples[n_layers-1])

        return log_p

    def log_prob_q(self, samples):
        """ Calculate q(h_{l+1} | h_l_ for all layers *but the first one*.  """
        n_layers = len(self.p_layers)
        n_samples = samples[0].shape[0]

        log_q = [None] * n_layers
        log_q[0] = tensor.zeros([n_samples])
        for l in xrange(n_layers-1):
            log_q[l+1] = self.q_layers[l].log_prob(samples[l+1], samples[l])

        return log_q

    #@application(inputs=['n_samples'], outputs=['samples', 'log_p', 'log_q'])
    def sample_p(self, n_samples):
        """
        """
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)
        
        samples = [None] * n_layers
        log_p = [None] * n_layers

        samples[n_layers-1], log_p[n_layers-1] = p_layers[n_layers-1].sample(n_samples)
        for l in reversed(xrange(1, n_layers)):
            samples[l-1], log_p[l-1] = p_layers[l-1].sample(samples[l])

        # Get log_q
        log_q = self.log_prob_q(samples)
    
        return samples, log_p, log_q

    #@application(inputs=['features'], 
    #             outputs=['samples', 'log_q', 'log_p'])
    def sample_q(self, features):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        batch_size = features.shape[0]
        
        samples = [None] * n_layers
        log_p = [None] * n_layers
        log_q = [None] * n_layers

        # Generate samples (feed-forward)
        samples[0] = features
        log_q[0] = tensor.zeros([batch_size])
        for l in xrange(n_layers-1):
            samples[l+1], log_q[l+1] = q_layers[l].sample(samples[l])

        # get log-probs from generative model
        log_p[n_layers-1] = p_layers[n_layers-1].log_prob(samples[n_layers-1])
        for l in reversed(range(1, n_layers)):
            log_p[l-1] = p_layers[l-1].log_prob(samples[l-1], samples[l])
            
        return samples, log_p, log_q

    #@application(inputs=['n_samples'], 
    #             outputs=['samples', 'log_q', 'log_p'])
    def sample(self, n_samples, oversample=100, n_inner=10):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        n_primary = n_samples*oversample

        samples, log_p, log_q = self.sample_p(n_primary)

        # Sum all layers
        log_p_all = sum(log_p)   # This is the python sum over a list
        log_q_all = sum(log_q)   # This is the python sum over a list

        _, log_qx = self.log_likelihood(samples[0], n_inner)

        log_w = (log_qx + log_q_all - log_p_all) / 2
        w_norm = logsumexp(log_w, axis=0)
        log_w = log_w-w_norm
        w = tensor.exp(log_w)

        #pvals = w.repeat(n_samples, axis=0)
        pvals = w.dimshuffle('x', 0).repeat(n_samples, axis=0)
        idx = self.theano_rng.multinomial(pvals=pvals).argmax(axis=1)

        subsamples = [s[idx,:] for s in samples]
    
        return subsamples, log_w

    @application(inputs=['log_p', 'log_q'], outputs=['w'])
    def importance_weights(self, log_p, log_q):
        # Sum all layers
        log_p_all = sum(log_p)   # This is the python sum over a list
        log_q_all = sum(log_q)   # This is the python sum over a list
    
        # Calculate sampling weights
        log_pq = (log_p_all-log_q_all)/2
        w_norm = logsumexp(log_pq, axis=1)
        log_w = log_pq-tensor.shape_padright(w_norm)
        w = tensor.exp(log_w)
        
        return w 
        

    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_psx'])
    def log_likelihood(self, features, n_samples):
        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)
        samples, log_p, log_q = self.sample_q(x)

        # Reshape and sum
        samples = unflatten_values(samples, batch_size, n_samples)
        log_p = unflatten_values(log_p, batch_size, n_samples)
        log_q = unflatten_values(log_q, batch_size, n_samples)

        log_p_all = sum(log_p)
        log_q_all = sum(log_q)

        # Approximate log(p(x))
        log_px  = logsumexp(log_p_all-log_q_all, axis=-1) - tensor.log(n_samples)
        log_psx = (logsumexp((log_p_all-log_q_all)/2, axis=-1) - tensor.log(n_samples)) * 2.

        return log_px, log_psx

    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_psx', 'gradients'])
    def get_gradients(self, features, n_samples):
        """Perform inference and calculate gradients.

        Returns
        -------
            log_px : T.fvector
            log_psx : T.fvector
            gradients : OrderedDict
        """
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)

        # Get Q-samples
        samples, log_p, log_q = self.sample_q(x)

        # Reshape and sum
        samples = unflatten_values(samples, batch_size, n_samples)
        log_p = unflatten_values(log_p, batch_size, n_samples)
        log_q = unflatten_values(log_q, batch_size, n_samples)

        log_p_all = sum(log_p)
        log_q_all = sum(log_q)

        # Approximate log p(x)
        log_px_bound = log_p_all[:,0] - log_q_all[:,0]
        log_px  = logsumexp(log_p_all-log_q_all, axis=-1) - tensor.log(n_samples)
        log_psx = (logsumexp((log_p_all-log_q_all)/2, axis=-1) - tensor.log(n_samples)) * 2.

        # Calculate IS weights
        w = self.importance_weights(log_p, log_q)

        wp = w.reshape( (batch_size*n_samples, ) )
        wq = w.reshape( (batch_size*n_samples, ) )
        wq = wq - (1./n_samples)

        samples = flatten_values(samples, batch_size*n_samples)

        gradients = OrderedDict()
        for l in xrange(n_layers-1):
            gradients = merge_gradients(gradients, p_layers[l].get_gradients(samples[l], samples[l+1], weights=wp))
            gradients = merge_gradients(gradients, q_layers[l].get_gradients(samples[l+1], samples[l], weights=wq))
        gradients = merge_gradients(gradients, p_layers[-1].get_gradients(samples[-1], weights=wp))

        if (self.l1reg > 0.) or (self.l2reg > 0.):
            reg_gradients = OrderedDict()
            params = Selector(self).get_parameters()
            for pname, param in params.iteritems():
                if has_roles(param, (WEIGHT,)):
                    reg_cost = self.l1reg * tensor.sum(abs(param)) + self.l2reg * tensor.sum(param**2)
                    reg_gradients[param] = tensor.grad(reg_cost, param)
            gradients = merge_gradients(gradients, reg_gradients)

        self.log_p_bound = log_px_bound
        self.log_p = log_px
        self.log_ph = log_psx

        return log_px, log_psx, gradients

    def estimate_log_z2(self, n_samples):
        """ Compute an estimate for 2log(z).

        Returns log(sum(sqrt( q(x|h)p(x,h') / (p(x,h)/q(h'|x)))))
            with x, h ~ p(x,h); h' ~ q(h|x)
        """
        samples, log_pp, log_pq = self.sample_p(n_samples)
        _, log_qp, log_qq = self.sample_q(samples[0])

        log_pp = sum(log_pp)
        log_pq = sum(log_pq)
        log_qp = sum(log_qp)
        log_qq = sum(log_qq)

        log_z2 = 1/2.*(log_pq-log_pp+log_qp-log_qq)
        log_z2 = logsumexp(log_z2)
        
        return log_z2
