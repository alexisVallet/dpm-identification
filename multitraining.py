""" Multi-class training
"""
import cv2
import numpy as np

import initmodel as init
import training as train

def binary_train(positives, negatives, feature, featdim,
                 mindimdiv, C=0.01, verbose=False):
    """ Full training procedure for the binary classification
        case, including initialization and LSVM training.

    Arguments:
        positives    positive image training samples.
        negatives    negative image training samples.
        feature      feature to use for feature maps.
        featdim      dimensionality of the feature vectors.
        mindimdiv    number of cells for the minimum dimension of
                     each image.
        C            (L)SVM soft-margin parameter.
    Returns:
        A mixture model for distinguishing positives and negatives.
    """
    # initialize the mixture model, and get the feature pyramids
    # as a byproduct.
    initmodel, pospyr, negpyr = init.initialize_model(
        positives,
        negatives,
        feature,
        featdim,
        mindimdiv=mindimdiv,
        C=C,
        verbose=verbose
    )

    print repr(initmodel)
    
    # run the training procedure on the initial model
    trainedmodel = train.train(
        initmodel,
        pospyr,
        negpyr,
        nbiter=4,
        C=C,
        verbose=verbose
    )

    # return the model
    return trainedmodel
