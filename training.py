""" Training algorithm for the dpm character identification method. 
"""
import numpy as np
import cv2
import sklearn as skl
import identify as idt
import dpm
import featpyramid as pyr
import matching

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
    Returns:
        a new, optimized model
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
    while gradient == None or np.linalg.norm(gradient) >= 10E-6:
        # set the learning rate
        alpha = 1 / t
        # choose a random example
        i = np.random.randint(0, nbexamples)
        # check whether it is positive or negative
        yi = 1 if i < nbpos else -1
        # If it is positive, pick the pre-set latent vector. It negative,
        # run the matching algorithm to find the best latent vector.
        latvec = None
        modellatdot = None
        if yi > 0:
            latvec = poslatents[:,i]
            modellatdot = np.vdot(currentbeta, latvec)
        else:
            # Get the latent vector and score from the matching
            # algorithm.
            (score, c, clatvec) = matching.mixture_matching(
                pyramid,
                dpm.vectortomixture(currentbeta, modelsize)
            )
            # make the latent vector for the entire mixture
            latvec = dpm.comp_latent_to_mixture(clatvec, modelsize, c,
                                                currentbeta.size)
            # the score of the DPM is none other than the dot product
            # between model and latent vectors.
            modellatdot = score
        # Compute the gradient from the sample, update beta and the model
        gradient = (currentbeta if yi * modellatdot >= 1 
                    else (beta - C * nbexamples * yi * latvec))
        previousbeta = np.copy(currentbeta)
        currentbeta = currentbeta - alpha * gradient

    return currentmodel

def train(initmodel, positives, negatives, nbiter=4, C=0.01):
    """ Trains a mixture of deformable part models using a latent SVM.

    Arguments:
        initmodel    initial mixture model
        positives    positive feature pyramid samples.
        negatives    negative feature pyramid samples.
        nbiter       the number of iterations of "coordinate descent"
                     to run.
        C            LSVM soft-margin parameter.

    Returns:
        A mixture model with the same size (i.e. number of components, number of
        parts for each component, parts sizes) as the initial model, optimized to
        classify positive and negative samples correctly.
    """
    currentmodel = initmodel
    vectorsize = initmodel.size().vectorsize()

    for t in range(0,nbiter):
        # First, compute the best latent values for each positive sample
        # using the matching algorithm
        poslatents = np.zeros([vectorsize, len(positives)])
        for pi in range(0, len(positives)):
            (score, c, latvec) = matching.mixture_matching(positives[pi],
                                                           currentmodel)
            poslatents[:,pi] = latvec
        
        # Then optimize the model using stochastic gradient descent
        currentmodel = lsvmsgd(currentmodel, poslatents, negatives, C)

    return currentmodel
