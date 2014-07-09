""" Implementation of the stochastic subgradient method to minimize convex
    non differentiable functions.
"""
import numpy as np

def ssm(init, samples, f_subgrad, nb_iter=50, nb_batches=None, 
        alpha_0=0.5, alpha_end=10E-5):
    """ Minimizes a convex non-differentiable function on a set of samples
        using the stochastic subgradient method.

    Arguments:
        init        initial n-dimensional numpy model vector to optimize.
        samples     array-like of samples.
        f_subgrad   subgradient of the function to minimize on a
                    subset of the input samples, e.g. 
                    f_subgrad(model, subset), where model is a n-dimensional
                    vector, and subset is an array-like of elements from the
                    input samples, should return a n-dimensional subgradient
                    of the function to minimize.
        nb_iter     number of iterations to run, e.g. number of time we go
                    through the entire dataset.
        nb_batches  number of batches to split the dataset into for subgradient
                    evaluation. If non specified or None, will default to the
                    number of input samples - we compute the subgradient for 
                    each sample individually. If 1, this essentially becomes
                    the non-stochastic subgradient method.
        alpha_0     initial learning rate.
        alpha_end   learning for the final iteration. The learning rate will
                    be decreased following a 1/t curve between alpha_0 and
                    alpha_end for each iteration.
    """
    # Check parameters.
    assert nb_iter >= 1
    if nb_batches == None:
        nb_batches = len(samples)
    assert 1 <= nb_batches <= len(samples)
    assert alpha_end <= alpha_0
    
    # Initialize stuff.
    model = np.array(init, copy=True)
    denom = (alpha_0 / alpha_end)**(1./(nb_iter - 1))
    alpha = alpha_0

    # Go crazy.
    for t in range(1, nb_iter):
        # Split the dataset into nb_batches roughly equally sized batches.
