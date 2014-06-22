""" Training algorithm for the dpm character identification method. 
"""
import numpy as np
import cv2
import sklearn as skl
import sklearn.preprocessing as sklpreproc
import identify as idt
import dpm
import featpyramid as pyr
import features as feat
import matching
import sgd

def lsvmsgd(model, negatives, poslatents, C, verbose=False):
    """ Latent svm stochastic gradient descent for optimizing the model
        given latent values for positive samples.

    Arguments:
        model      initial model.
        poslatents m by n matrix where n is the number of positive 
                   samples, and column i contains the latent vector 
                   for the ith positive example.
        negatives  feature pyramids for all negative examples.
        C          soft-margin parameter for the SVM.
    Returns:
        a new, optimized model
    """
    # the model size is unchanged by training, and we need it to
    # convert from vector representation to the (more convenient)
    # Mixture objects.
    modelsize = model.size()
    nb_pos = poslatents.shape[1]
    nb_samples = nb_pos + len(negatives)
    init = model.tovector()
    # keeping track of percentages of matching for each component
    negcomps = np.zeros([len(model.dpms)], np.float32)

    # gradient computation closure
    def gradient(weights, i):
        """ Computes the gradient for sample i at the point specified
            by the weights vector.
        """
        yi = 1. if i < nb_pos else -1.
        latvec = None
        if yi > 0:
            # If the sample is positive, then get the precomputed latent
            # vector and compute fb from it.
            latvec = poslatents[:,i]
        else:
            # If the sample is negative, find the best latent vector
            # using the matching algorithm.
            (score,compidx,latvec_) = matching.mixture_matching(
                negatives[i - nb_pos],
                dpm.vectortomixture(weights, modelsize)
            )
            # keep track of negative matchings
            negcomps[compidx] += 1
            latvec = latvec_
        fb = np.vdot(weights, latvec)
        hi = None
        if yi * fb >= 1:
            hi = 0
        else:
            hi = -yi * latvec
        return weights + C * float(nb_samples) * hi
    
    # run stochastic gradient descent
    final = sgd.sgd(nb_samples, init, gradient, verbose=verbose)
    
    # return the corresponding model
    return (dpm.vectortomixture(final, modelsize), 
            negcomps/negcomps.sum())

def train(initmodel, positives, negatives, nbiter=4, C=0.01, verbose=False):
    """ Trains a mixture of deformable part models using a latent SVM.

    Arguments:
        initmodel    initial mixture model
        positives    positive feature pyramid samples.
        negatives    negative feature pyramid samples.
        nbiter       the number of iterations of "coordinate descent"
                     to run.
        C            LSVM soft-margin parameter.

    Returns:
        A mixture model with the same size (i.e. number of components, 
        number of parts for each component, parts sizes) as the initial 
        model, optimized to classify positive and negative samples 
        correctly.
    """
    currentmodel = initmodel
    vectorsize = initmodel.size().vectorsize()

    for t in range(0,nbiter):
        # First, compute the best latent values for each positive sample
        # using the matching algorithm
        print "computing latent values for positive samples..."
        poslatents = np.zeros([vectorsize, len(positives)])
        poscomps = np.zeros([len(currentmodel.dpms)], np.float32)
        for pi in range(0, len(positives)):
            (score, c, latvec) = matching.mixture_matching(positives[pi],
                                                           currentmodel)
            poslatents[:,pi] = latvec
            poscomps[c] += 1
        
        # Then optimize the model using stochastic gradient descent
        print "running gradient descent to optimize the mixture..."
        (currentmodel, negcomps) = lsvmsgd(
            currentmodel, negatives, poslatents, C,
            verbose=verbose)
        if verbose:
            poscomps /= len(positives)
            print ("positive component matches: " + 
                   repr(poscomps))
            print "negative component matches: " + repr(negcomps)
            print ("total component matches: " + 
                   repr((poscomps + negcomps)/2.))

    return currentmodel
