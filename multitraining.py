""" Multi-class training
"""
import cv2
import numpy as np
import multiprocessing as mp
import math
import sys
import os

import features as feat
import bintraining as btrain

# awful, awful hack to avoid the training data to be copied
# in an extremely inefficient manner (pickling) for each
# worker thread.
global_traindata = None

def runbintrain(arguments):
    """ Runs training for a single class label.
    """
    # because multiprocessing sucks balls
    label, mindimdiv, C, verbosity = arguments
    nbbins = (4,4,4)
    feature = feat.bgrhistogram(nbbins)
    featdim = np.prod(nbbins)

    if verbosity > 0:
        print "running training for " + repr(label) + "..."
    positives = global_traindata[label]
    negatives = reduce(lambda l1, l2: l1 + l2,
                       map(lambda k: global_traindata[k],
                           [k for k in global_traindata if k != label]))
    model = btrain.binary_train(positives, negatives, feature, featdim,
                                mindimdiv, C, verbosity > 1)
    if verbosity > 0:
        print "finished training for " + repr(label)

    return model

def multi_train(traindata, mindimdiv, C=0.01, verbosity=0):
    """ Trains a model for multi-class classification using deformable
        parts models. In practice, trains n binary classifiers in a one vs
        all fashion in parallel.

    Arguments:
        traindata    dictionary from class labels to corresponding sets
                     of images.
        feature      feature function to use for building feature maps.
        featdim      dimensionality of the vectors returned by the feature
                     function.
        mindimdiv    number of times to divide the smallest dimension of
                     each image to build feature maps.
        C            (L)SVM soft-margin parameter.
        verbosity    set to 0 for no messages, 1 for few messages (little
                     overhead) and 2 for a lot of messages (big overhead,
                     debug only).
    """
    # Passing the whole training data as argument to runbintrain causes
    # the data to be send by pickling to the subprocesses. Which is
    # horribly, horribly inefficient. Putting it in a global variable
    # causes it to be passed with the OS process forking semantic, which
    # in the case of linux with cpython means shared address space and
    # copy on write. Which is awesome. Yet horrible.
    global global_traindata 
    global_traindata = traindata
    pool  = mp.Pool()
    labels = [k for k in traindata]

    # run each batch on its own process
    # had to explicitely curry it at the top level so it didn
    arguments = map(lambda k: (k, mindimdiv, C, verbosity),
                    labels)
    models = pool.map(runbintrain, arguments)

    results = {}

    for i in range(len(labels)):
        results[labels[i]] = models[i]

    return results

if __name__ == "__main__":
    # command line interface to training
    # requires an image folder. Image suffixes are taken as
    # class labels.
    if len(sys.argv) < 2:
        raise ValueError("Please input a folder of training images.")
    
