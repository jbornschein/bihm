
from __future__ import division, print_function 

import sys
import logging

import numpy
import theano

from theano import tensor

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks import Random, Initializable, MLP, Logistic
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity

from . import HelmholtzMachine, replicate_batch, logsumexp
from .prob_layers import GaussianLayer, BernoulliLayer
from .initialization import RWSInitialization

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

#-----------------------------------------------------------------------------


class VAE(HelmholtzMachine, Random):
    """ Variational Autoencoder with
            p(z) = Gaussian( ... )  and
            p(x|z) = Bernoulli( ... )
    """
    def __init__(self, x_dim, hidden_layers, hidden_act, z_dim, batch_norm=False, **kwargs):
        super(VAE, self).__init__([], [], **kwargs)

        inits = {
            'weights_init': IsotropicGaussian(std=0.1),
            #'weights_init': RWSInitialization(factor=1.),
            'biases_init': Constant(0.0),
        }

        hidden_act = [hidden_act]*len(hidden_layers)

        assert batch_norm == False
        q_mlp = MLP(hidden_act             , [x_dim]+hidden_layers, **inits)
        p_mlp = MLP(hidden_act+[Logistic()], [z_dim]+hidden_layers+[x_dim], **inits)

        self.q = GaussianLayer(z_dim, q_mlp, **inits)
        self.p = BernoulliLayer(p_mlp, **inits)

        self.prior_log_sigma = numpy.zeros(z_dim)    # 
        self.prior_mu = numpy.zeros(z_dim)           # 

        self.children = [self.p, self.q]


    @application(inputs=['n_samples'], outputs=['samples'])
    def sample(self, n_samples):
        """ Sample from the generative mdoel """
        # Sample from mean-zeros std.-one Gaussian
        eps = self.theano_rng.normal(
                    size=(n_samples, self.dim_X),
                    avg=0., std=1.)

        # ... and scale/translate samples
        z = self.prior_mu + tensor.exp(seld.prior_log_sigma) * eps
 
        x, _ = self.p.sample(z)

        return [x, z]


    @application(inputs=['features'], outputs=['samples', 'log_p', 'log_q'])
    def sample_q(self, features):
        """ Sample from the approx inference network Q """
        z, log_q = self.q.sample(features)
        log_p = self.p.log_prob(features, z)     # p(x|z)
        log_p += tensor.sum(                     # p(z) prior
            -0.5*tensor.log(2*numpy.pi)
            -self.prior_log_sigma
            -0.5*(z-self.prior_mu)**2 / tensor.exp(2*self.prior_log_sigma), axis=1)

        return [z], [log_p], [log_q]


    @application(inputs=['log_p', 'log_q'], outputs=['w'])
    def importance_weights(self, log_p, log_q):
        """ Calculate importance weights for the given samples """

        # Sum all layers
        log_p_all = sum(log_p)   # Python sum over a list
        log_q_all = sum(log_q)   # Python sum over a list

        # Calculate sampling weights
        log_pq = (log_p_all-log_q_all)
        w_norm = logsumexp(log_pq, axis=1)
        log_w = log_pq-tensor.shape_padright(w_norm)
        w = tensor.exp(log_w)

        return w


    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_px'])
    def log_likelihood(self, features, n_samples):
        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)
        samples, log_p, log_q = self.sample_q(x)
        z = samples[0]
        log_p = log_p[0]
        log_q = log_q[0]

        log_q = log_q.reshape([batch_size, n_samples])
        log_p = log_p.reshape([batch_size, n_samples])
        log_px = logsumexp(log_p-log_q, axis=1) - tensor.log(n_samples)

        return log_px, log_px


    @application(inputs=['features', 'n_samples'], outputs=['log_p_bound'])
    def log_likelihood_bound(self, features, n_samples=1):
        """Compute the LL bound. """
        batch_size = features.shape[0]

        z_mu, z_log_sigma = self.q.sample_expected(features)

        # Recosntruction...
        features_r    = replicate_batch(features, n_samples)
        z_mu_r        = replicate_batch(z_mu, n_samples)
        z_log_sigma_r = replicate_batch(z_log_sigma, n_samples)

        epsilon = self.theano_rng.normal(size=z_mu_r.shape, dtype=z_mu_r.dtype)
        z_r = z_mu_r + epsilon * tensor.exp(z_log_sigma_r)

        recons_term = self.p.log_prob(features_r, z_r)
        recons_term = recons_term.reshape([batch_size, n_samples])
        recons_term = tensor.sum(recons_term, axis=1) / n_samples

        # KL divergence
        per_dim_kl = (
                self.prior_log_sigma - z_log_sigma 
                + (tensor.exp(2*z_log_sigma)+(z_mu-self.prior_mu)**2) / tensor.exp(2*self.prior_log_sigma) / 2.
                - 0.5)
    
        kl_term = per_dim_kl.sum(axis=1)

        self.kl_term = kl_term
        self.kl_term.name = 'kl_term'
        self.recons_term = recons_term
        self.recons_term.name = 'recons_term'

        log_p_bound = recons_term - kl_term

        return log_p_bound
