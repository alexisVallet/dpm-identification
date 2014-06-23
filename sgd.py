""" Implementation of stochastic gradient descent.
"""
import numpy as np

def sgd(nb_samples, init, gradient, t0=2.0, nbiter=5, verbose=False):
    """ Minimizes a function using stochastic gradient descent on a
        dataset.
    
    Arguments:
        nb_samples number of samples.
        init       initial nb_features dimensional vector to optimize
        gradient   gradient of the function to optimize for a single
                   sample. e.g. gradient(model, i) where model is
                   the current vector to optimize and i is the randomly
                   chosen sample. must return a nb_features dimensional 
                   gradient vector.
        regr       regularization parameter used to update the learning 
                   rate alpha, e.g at iteration t, alpha = 1 / (regr * t). 
                   Smaller values increase the runtime efficiency, but may 
                   not make the algorithm converge.
        nbiter     maximum number of iterations.
        verbose    if true, will print information messages

    Returns:
        An nb_features dimensional vector that minimizes the function.
    """
    t = t0
    weights = init
    gradient_ = None

    # run nbiter epochs
    for epoch in range(nbiter):
        if verbose:
            print "running epoch " + repr(epoch) + "..."
        # randomly shuffle the data
        randidxs = np.random.permutation(nb_samples)
        # keep track of the average gradient norm across the epoch
        sumgradient = 0

        # for each sample in the shuffled data, compute a subgradient
        # and update the model accordingly
        for i in randidxs:
            # compute the gradient
            gradient_ = gradient(weights, i)
            sumgradient += np.linalg.norm(gradient_)
            # update the learning rate
            alpha = 1.0 / t
            # compute the new weights
            weights = weights - alpha * gradient_
            # increase the iteration counter
            t += 1
        if verbose:
            print "avg gradient norm: " + repr(sumgradient / nb_samples)
            print "learning rate: " + repr(alpha)
            print "weights norm: " + repr(np.linalg.norm(weights))
    
    return weights