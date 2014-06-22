""" Implementation of stochastic gradient descent.
"""
import numpy as np

def sgd(nb_samples, init, gradient, regr=0.001, nbiter=5, verbose=False):
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
    # code to initialize the learning rate from scikit-learn (not sure
    # what the theory is behind it to be honest, but whatever)
    typw = np.sqrt(1.0 / np.sqrt(regr))
    eta0 = typw / (1.0 + typw)
    t = 1.0 / (eta0 * regr)
    if verbose:
        print "t0 = " + repr(t)
    weights = init
    gradient_ = None

    # run nbiter epochs
    for epoch in range(nbiter):
        if verbose:
            print "running epoch " + repr(epoch) + "..."
        # randomly shuffle the data
        randidxs = np.random.permutation(nb_samples)

        # for each sample in the shuffled data, compute a subgradient
        # and update the model accordingly
        for i in randidxs:
            # compute the gradient
            gradient_ = gradient(weights, i)
            # update the learning rate
            alpha = 1.0 / (regr * t)
            # compute the new weights
            weights = weights - alpha * gradient_
            # increase the iteration counter
            t += 1
        if verbose:
            print "gradient norm: " + repr(np.linalg.norm(gradient_))
            print "learning rate: " + repr(alpha)
            print "weights norm: " + repr(np.linalg.norm(weights))
    
    return weights
