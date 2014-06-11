""" Training algorithm for the dpm character identification method. 
"""
import numpy as np
import cv2
import sklearn as skl
import identify as idt
import dpm
import featpyramid as pyr

def bestlatvec(model, pyramid):
    """ Compute the best latent vector (the one corresponding to the highest 
        scoring object hypothesis) for a model by matching it against a specific 
        pyramid.
    """
        

def lsvmsgd(model, poslatents, negatives, C):
    """ Latent svm stochastic gradient descent for optimizing the model
        given latent values for positive samples.

    Arguments:
        model      initial model.
        poslatents m by n matrix where n is the number of positive samples,
                   and column i contains the latent vector for the ith positive
                   example.
        negatives  feature pyramids for all negative examples.
        C          soft-margin parameter for the SVM.
    """
    previousbeta = None
    currentbeta = model.tovector()
    modelsize = model.size()
    nbpos = poslatents.shape[1]
    nbneg = len(negatives)
    nbexamples = nbpos + nbneg
    t = 1
    gradient = None
    
    # Stop when the gradient is too small
    while gradient == None or np.linalg.norm(gradient) >= 10E-8  :
        # set the learning rate
        alpha = 1 / t
        # choose a random example
        i = np.random.randint(0, nbexamples)
        # check whether it is positive or negative
        yi = 1 if i < nbpos else -1
        # If it is positive, pick the pre-set latent vector. It negative,
        # run the matching algorithm to find the best latent vector.
        latvec = poslatents[:,i] if yi > 0 else bestlatvec(vectortomixture(currentbeta.
                                                                           modelsize),
                                                           negatives[i-nbpos])
        # Compute the gradient from the sample, update beta and the model
        gradient = (currentbeta if yi * (currentbeta.vdot(latvec)) >= 1 
                    else (beta - C * nbexamples * yi * latvec))
        previousbeta = np.copy(currentbeta)
        currentbeta = currentbeta - alpha * gradient

    return currentmodel

def train(initmodel, positives, negatives):
    """ Trains a mixture of deformable part models using a latent SVM.

    Arguments:
        initmodel    initial mixture model
        positives    positive feature pyramid samples.
        negatives    negative feature pyramid samples.

    Returns:
        A mixture model with the same size (i.e. number of components, number of
        parts for each component, parts sizes) as the initial model, optimized to
        classify positive and negative samples correctly.
    """
    # First, compute latent values for each 
    
