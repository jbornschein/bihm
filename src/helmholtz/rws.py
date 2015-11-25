
from __future__ import division, print_function

import sys
sys.path.append("../")

import logging

import numpy
import theano

from theano import tensor
from collections import OrderedDict

from blocks.bricks.base import application, Brick, lazy
from blocks.select import Selector

from . import HelmholtzMachine
from . import flatten_values, unflatten_values, merge_gradients, replicate_batch, logsumexp

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

#-----------------------------------------------------------------------------


class ReweightedWakeSleep(HelmholtzMachine):
    def __init__(self, p_layers, q_layers, **kwargs):
        super(ReweightedWakeSleep, self).__init__(p_layers, q_layers, **kwargs)

    def log_prob_p(self, samples):
        """Calculate p(h_l | h_{l+1}) for all layers. """
        n_layers = len(self.p_layers)
        n_samples = samples[0].shape[0]

        log_p = [None] * n_layers
        for l in xrange(n_layers-1):
            log_p[l] = self.p_layers[l].log_prob(samples[l], samples[l+1])
        log_p[n_layers-1] = self.p_layers[n_layers-1].log_prob(samples[n_layers-1])

        return log_p

    def log_prob_q(self, samples):
        """Calculate q(h_{l+1} | h_l) for all layers *but the first one*. """
        n_layers = len(self.p_layers)
        n_samples = samples[0].shape[0]

        log_q = [None] * n_layers
        log_q[0] = tensor.zeros([n_samples])
        for l in xrange(n_layers-1):
            log_q[l+1] = self.q_layers[l].log_prob(samples[l+1], samples[l])

        return log_q

    @application(inputs=['n_samples'], outputs=['samples', 'log_p', 'log_q'])
    def sample_p(self, n_samples):
        """Samples form the prior.
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

    @application(inputs=['features'], outputs=['samples', 'log_p', 'log_q'])
    def sample_q(self, features):
        """Sample from q(h|x).

        Parameters
        ----------
        features : Tensor

        Returns
        -------
        samples : list
        log_p : list
        log_q : list
        """
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

    @application(inputs=['n_samples'], outputs=['samples', 'log_p', 'log_q'])
    def sample(self, n_samples):
        return self.sample_p(n_samples)

    @application(inputs=['log_p', 'log_q'], outputs=['w'])
    def importance_weights(self, log_p, log_q):
        """ Calculate importance weights for the given samples """

        # Sum all layers
        log_p_all = sum(log_p)   # This is the python sum over a list
        log_q_all = sum(log_q)   # This is the python sum over a list

        # Calculate sampling weights
        log_pq = (log_p_all-log_q_all)
        w_norm = logsumexp(log_pq, axis=1)
        log_w = log_pq-tensor.shape_padright(w_norm)
        w = tensor.exp(log_w)

        return w

    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_psx'])
    def log_likelihood(self, features, n_samples):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

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

        return log_px, log_px

    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_psx', 'gradients'])
    def get_gradients(self, features, n_samples):
        """Perform inference and calculate gradients.

        Returns
        -------
            log_px    : T.fvector
            log_psx   : T.fvector
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

        # Approximate log p(x) and calculate IS weights
        w = self.importance_weights(log_p, log_q)

        # Approximate log(p(x))
        log_px  = logsumexp(log_p_all-log_q_all, axis=-1) - tensor.log(n_samples)

        w = w.reshape( (batch_size*n_samples, ) )
        samples = flatten_values(samples, batch_size*n_samples)

        gradients = OrderedDict()
        for l in xrange(n_layers-1):
            gradients = merge_gradients(gradients, p_layers[l].get_gradients(samples[l], samples[l+1], weights=w))
            gradients = merge_gradients(gradients, q_layers[l].get_gradients(samples[l+1], samples[l], weights=w), 0.5)
        gradients = merge_gradients(gradients, p_layers[-1].get_gradients(samples[-1], weights=w))

        # Now sleep phase..
        samples, log_p, log_q = self.sample_p(batch_size)
        for l in xrange(n_layers-1):
            gradients = merge_gradients(gradients, q_layers[l].get_gradients(samples[l+1], samples[l]), 0.5)

        return log_px, log_px, gradients
