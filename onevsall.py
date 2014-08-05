""" Implementation of parallel one vs all classification from a binary
    probabilistic image classifier.
"""
import numpy as np
import multiprocessing as mp
import os
import cPickle as pickle
import time

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
    starttime = time.time()
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
    
    endtime = time.time()
    if verbose:
        print ("Finished training for " + repr(label) + " in " 
               + repr(endtime - starttime) + " seconds.")

    # Cache if possible.
    if cachedir != None:
        cachefile = open(cachefilename, 'w')
        pickle.dump(binclassifier, cachefile)
        cachefile.close()

    return binclassifier

class OneVSAll:
    def __init__(self, binclassinit, cachedir=None, nb_cores=None, verbose=False):
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
        self.nb_cores = nb_cores

    def train(self, samples, labels):
        """ Trains a multi-class classifier from labeled samples. Runs
            training in a one-vs-all fashion, training individual binary
            classifiers in parallel using as many CPU cores as available.
        
        Arguments:
            samples    array-like of nb_samples training samples.
            labels     array-like of nb_samples labels corresponding
                       to the samples.
        """
        starttime = time.time()
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
        pool = mp.Pool(processes=self.nb_cores)
        self.labels_set = list(set(labels))
        binmodels = pool.map(_binary_train, self.labels_set)
        self.binmodels = binmodels
        _onevall = None
        endtime = time.time()
        if self.verbose:
            print ("Trained " + repr(len(self.labels_set)) + 
                   " classifiers in " + repr(endtime - starttime) 
                   + " seconds.")

    def predict_labels(self, samples):
        """ Predicts labels for test samples.

        Arguments:
            samples
                list of sample to classify.
        
        Returns:
            list of corresponding class labels.
        """
        # Run each classifier on each sample. Pick the
        # highest probability for each sample.
        nb_samples = len(samples)
        nb_classes = len(self.labels_set)
        probas = np.empty([nb_samples, nb_classes])

        for j in range(len(self.labels_set)):
            probas[:,j] = self.binmodels[j].predict_proba(samples)
        labelidxs = np.argmax(probas, 1)
        print probas.shape
        print labelidxs.shape
        print len(self.labels_set)

        return map(lambda idx: self.labels_set[idx], labelidxs)
