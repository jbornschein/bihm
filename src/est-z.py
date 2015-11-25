#!/usr/bin/env python 

from __future__ import print_function, division

import sys
sys.path.append("..")
sys.setrecursionlimit(100000)

import os
import logging

import numpy as np
import cPickle as pickle

import theano
import theano.tensor as tensor

from PIL import Image
from argparse import ArgumentParser
from progressbar import ProgressBar
from scipy import stats

from blocks.main_loop import MainLoop

import scipy.misc as misc

import helmholtz.datasets as datasets

from helmholtz import logsumexp
from helmholtz.bihm import BiHM
from helmholtz.gmm import GMM
from helmholtz.rws import ReweightedWakeSleep

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)


#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser("Estimate the effective sample size")
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--nsamples", "--samples", "-s", type=int, 
            default=1000000, help="no. of samples to draw")
    parser.add_argument("--ninner", type=int, 
            default=10, help="no. of samples to draw")
    parser.add_argument("--batch-size", "-bs", type=int, 
            default=10000, help="no. of samples to draw")
    parser.add_argument("experiment", help="Experiment to load")
    args = parser.parse_args()

    logger.info("Loading model %s..." % args.experiment)
    with open(args.experiment, "rb") as f:
        m = pickle.load(f)

    if isinstance(m, MainLoop):
        m = m.model

    brick = m.get_top_bricks()[0]
    while len(brick.parents) > 0:
        brick = brick.parents[0]

    assert isinstance(brick, (ReweightedWakeSleep, BiHM, GMM))

    np.random.seed();
    for layer in brick.p_layers:
        layer.theano_rng.seed(np.random.randint(500))

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    np.random.seed(999)

    batch_size = 1
    n_samples = tensor.iscalar('n_samples')
    n_inner = tensor.iscalar('n_inner')
    #x = tensor.matrix('features')
    #x_ = replicate_batch(x, n_samples)

    samples, log_p, log_q = brick.sample_p(n_samples)
    log_px, log_psx = brick.log_likelihood(samples[0], n_inner)

    log_p = sum(log_p)
    log_q = sum(log_q)

    log_psxp  = 1/2.*log_psx + 1/2.*(log_q-log_p)
    log_psxp2 = 2 * log_psxp

    log_psxp  = logsumexp(log_psxp)
    log_psxp2 = logsumexp(log_psxp2)

    do_z = theano.function(
                        [n_samples, n_inner], 
                        [log_psxp, log_psxp2],
                        name="do_z", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Computing Z...")

    batch_size = args.batch_size // args.ninner

    n_samples = []
    log_psxp  = []
    log_psxp2 = []
    for k in xrange(0, args.nsamples, batch_size):
        psxp, psxp2 = do_z(batch_size, args.ninner)
        psxp, psxp2 = float(psxp), float(psxp2)

        n_samples.append(k)
        log_psxp.append(psxp)
        log_psxp2.append(psxp2)

        if k % 10000 == 0:
            sum_psxp = misc.logsumexp(log_psxp)
            sum_psxp2 = misc.logsumexp(log_psxp2)

            z_est = (sum_psxp - np.log(k)) / 2
            sd = np.sqrt(k*np.exp(sum_psxp)-np.exp(sum_psxp2)) / k
            se = 2*sd / np.sqrt(k)

            print("[%d samples] Z (p*) estimate: %7.5f +- %7.5f" % (k, z_est, se))

    import pandas as pd
    df = pd.DataFrame({'k': n_samples, 'log_psxp': log_psxp, 'log_psxp2': log_psxp2})
    df.save("est-Z-inner%d.pkl" % (args.ninner))
