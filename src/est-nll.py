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
from scipy.misc import logsumexp

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten

from blocks.main_loop import MainLoop

import helmholtz.datasets as datasets

from helmholtz import replicate_batch
from helmholtz.gmm import GMM
from helmholtz.bihm import BiHM
from helmholtz.rws import ReweightedWakeSleep
from helmholtz.vae import VAE

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser("Estimate the effective sample size")
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--max-batch", type=int, 
            default="10000", help="Maximum internal batch size (default: 10000)")
    parser.add_argument("--nsamples", "--samples", "-s", type=str, 
            default="1,10,100,1000,10000", help="Comma seperated list of #samples")
    parser.add_argument("--no-z-est", "-noz", action="store_true", default=False,
            help="Do not estimate log Z for BiHM models")
    parser.add_argument("--zsamples", type=int, default=1000000,
            help="Estimate Z using this number of samples")
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

    assert isinstance(brick, (ReweightedWakeSleep, GMM, BiHM, VAE))

    #----------------------------------------------------------------------
    estimate_z = not args.no_z_est and isinstance(brick, (BiHM, GMM))
    if estimate_z:
        logger.info("Estimating log z...")

        # compile theano function
        bs = tensor.iscalar('bs')
        log_z2 = brick.estimate_log_z2(bs)

        do_z = theano.function(
            [bs],
            log_z2,
            name="do_z", allow_input_downcast=True)

        #-------------------------------------------------------

        seq = []
        pbar = ProgressBar()
        for _ in pbar(xrange(0, args.zsamples, args.max_batch)):
            seq.append(float(do_z(args.max_batch)))
        
        log_z2 = logsumexp(seq) - np.log(args.zsamples)
                
        logger.info("2 log z ~= %5.3f" % log_z2)

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    n_samples = tensor.iscalar('n_samples')
    x = tensor.matrix('features')

    log_p, log_ps = brick.log_likelihood(x, n_samples)
    
    do_nll = theano.function(
                        [x, n_samples], 
                        [log_p, log_ps],
                        name="do_nll", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Loading dataset...")

    x_dim, data_train, data_valid, data_test = datasets.get_data(args.data)

    num_examples = data_test.num_examples
    n_samples = (int(s) for s in args.nsamples.split(","))

    dict_p = {}
    dict_ps = {}
    
    for K in n_samples:
        batch_size = max(args.max_batch // K, 1)
        stream = Flatten(DataStream(
                        data_test,
                        iteration_scheme=ShuffledScheme(num_examples, batch_size)
                    ), which_sources='features')

        log_p = np.asarray([])
        log_ps = np.asarray([])
        for batch in stream.get_epoch_iterator(as_dict=True):
            log_p_, log_ps_ = do_nll(batch['features'], K)
    
            log_p = np.concatenate((log_p, log_p_))
            log_ps = np.concatenate((log_ps, log_ps_))
    
        log_p_ = stats.sem(log_p)
        log_p = np.mean(log_p)
        log_ps_ = stats.sem(log_ps)
        log_ps = np.mean(log_ps)

        dict_p[K] = log_p
        dict_ps[K] = log_ps
    
        if estimate_z:
            print("log p / log p~ / log p* [%6d spls]:  %5.2f+-%4.2f  /  %5.2f+-%4.2f  /  %5.2f" % 
                (K, log_p, log_p_, log_ps, log_ps_, log_ps-log_z2))
        else:
            print("log p / log p~ [%6d spls]:  %5.2f+-%4.2f  /  %5.2f+-%4.2f" %
                (K, log_p, log_p_, log_ps, log_ps_))
