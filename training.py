""" Training algorithm for the dpm character identification method. 
"""
import numpy as np
import cv2
import sklearn as skl
import identify as idt
import dpm
import featpyramid as pyr
import features as feat
import matching

def lsvmsgd(model, poslatents, negatives, C, maxiter=None):
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
    currentbeta = model.tovector()
    modelsize = model.size()
    nbpos = poslatents.shape[1]
    nbneg = len(negatives)
    nbexamples = nbpos + nbneg
    if maxiter == None:
        maxiter = nbexamples
    t0 = 2
    diff = None
    
    t = t0
    # Stop after a given number of iterations
    while t < maxiter + t0:
        # shuffle the training data
        tidxs = np.random.permutation(nbexamples)
        
        for i in tidxs:
            # set the learning rate
            alpha = 1. / t
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
                (score, c, latvec) = matching.mixture_matching(
                    negatives[i-nbpos],
                    dpm.vectortomixture(currentbeta, modelsize)
                )
                # the score of the DPM is none other than the dot product
                # between model and latent vectors.
                modellatdot = score
            # Compute the gradient from the sample, update beta and the model
            gradient = (currentbeta if yi * modellatdot >= 1 
                        else (currentbeta - C * nbexamples * yi * latvec))
            diff = alpha * gradient
            # If we're not making any progress (either because of the
            # learning rate or the gradient being to small) we stop.
            if np.linalg.norm(diff) <= 10E-8:
                print "reached a minimum at t = " + repr(t - t0)
                print "gradient = " + repr(np.linalg.norm(gradient))
                print "lrate = " + repr(alpha)
                return dpm.vectortomixture(currentbeta, modelsize)
            currentbeta = currentbeta - diff
            t = t + 1

    # If we have reached too many iterations, return the current model
    print "reached maximum iteration"
    return dpm.vectortomixture(currentbeta, modelsize)

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
        # debug: show how the model evolves
        i = 0
        for comp in currentmodel.dpms:
            rootimg = feat.visualize_featmap(comp.root, feat.bgrhistvis((4,4,4)))
            winname = "t = " + repr(t) + " root " + repr(i)
            cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
            cv2.imshow(winname, rootimg)
        cv2.waitKey(0)
        # First, compute the best latent values for each positive sample
        # using the matching algorithm
        print "computing latent values for positive samples..."
        poslatents = np.zeros([vectorsize, len(positives)])
        for pi in range(0, len(positives)):
            (score, c, latvec) = matching.mixture_matching(positives[pi],
                                                           currentmodel)
            poslatents[:,pi] = latvec
        
        # Then optimize the model using stochastic gradient descent
        print "running gradient descent to optimize the mixture..."
        currentmodel = lsvmsgd(currentmodel, poslatents, negatives, C)

    return currentmodel
