""" Implementation of parallel one vs all classification from a binary
    probabilistic image classifier.
"""
import numpy as np
import multiprocessing as mp
import os
import cPickle as pickle

import RemoteException

# Global variable to avoid the expensive pickling from multiprocessing,
# and benefit from Linux's copy on write semantics.
_onevall = None

@RemoteException.showError
def _binary_train(label):
    """ Trains and returns a binary classifier for the given class
        label.
    """
    verbose = _onevall['verbose']
    if verbose:
        print "Training for " + repr(label) + "..."
    global _onevall
    # Check the cache.
    cachedir = _onevall['cachedir']
    cachefilename = None
    if cachedir != None:
        cachefilename = os.path.join(cachedir, repr(label))
        # If cached file exists, load and return it.
        if os.path.isfile(cachefilename):
            cachefile = open(cachefilename)
            binmodel = pickle.load(cachefile)
            cachefile.close()
            return binmodel

    # Build positive and negative sets for the given label.
    samples = _onevall['samples']
    labels = _onevall['labels']
    positives = [samples[i] for i in range(len(samples))
                 if labels[i] == label]
    negatives = [samples[i] for i in range(len(samples))
                 if labels[i] != label]
    # Run binary training.
    binclassinit = _onevall['binclassinit']
    binclassifier = binclassinit()
    binclassifier.train(positives, negatives)
    
    # Cache if possible.
    if cachedir != None:
        cachefile = open(cachefilename, 'w')
        pickle.dump(binclassifier, cachefile)
        cachefile.close()

    if verbose:
        print "Finished training for " + repr(label) + "."

    return binclassifier

class OneVSAll:
    def __init__(self, binclassinit, cachedir=None, verbose=False):
        """ Initializes the one-vs-all classifier given an instance
            of a binary classifier.
        
        Argument:
            binclassinit    function for instantiating a binary 
                            classifier. Should not take any
                            parameter. The returned classifier should
                            be picklable after being trained.
            cachedir        directory to cache intermediate data to.
                            If None, no caching will be performed.
            verbose         set to true for regular information messages
                            to be printed.
        """
        self.binclassinit = binclassinit
        self.cachedir = cachedir
        self.verbose = verbose

    def train(self, samples, labels):
        """ Trains a multi-class classifier from labeled samples. Runs
            training in a one-vs-all fashion, training individual binary
            classifiers in parallel using as many CPU cores as available.
        
        Arguments:
            samples    array-like of nb_samples training samples.
            labels     array-like of nb_samples labels corresponding
                       to the samples.
        """
        # Passing most of the parameters via global variable to avoid
        # pickling, both for performance (much more efficient copy of
        # training data) and for better interface (binary classifier
        # instantiation does not have to be picklable).
        global _onevall
        _onevall = {
            'samples': samples,
            'labels': labels,
            'binclassinit': self.binclassinit,
            'cachedir': self.cachedir,
            'verbose': self.verbose
        }
        # Running the training of each binary classifier on a pool of
        # parallel processes.
        pool = mp.Pool()
        binmodels = pool.map(_binary_train, set(labels))
        self.binmodels = binmodels
        _onevall = None
