""" Implementation of the stochastic subgradient method to minimize convex
    non differentiable functions.
"""
import numpy as np

def ssm(init, samples, f_subgrad, nb_iter=50, nb_batches=None, 
        alpha_0=0.8, alpha_end=10E-5, verbose=False):
    """ Minimizes a convex non-differentiable function on a set of samples
        using the stochastic subgradient method.

    Arguments:
        init        initial n-dimensional numpy model vector to optimize.
        samples     array-like of samples.
        f_subgrad   subgradient of the function to minimize on a
                    subset of the input samples, e.g. 
                    f_subgrad(nb_batches, model, subset), where model is 
                    a n-dimensional vector, and subset is an array-like of
                    elements from the input samples, should return a 
                    n-dimensional subgradient of the function to minimize.
        nb_iter     number of iterations to run, e.g. number of time we go
                    through the entire dataset.
        nb_batches  number of batches to split the dataset into for 
                    subgradient evaluation. If non specified or None, will
                    default to batches of size 10. If 1, this essentially
                    becomes the non-stochastic subgradient method.
        alpha_0     initial learning rate.
        alpha_end   learning for the final iteration. The learning rate 
                    will be decreased following a 1/t curve between 
                    alpha_0 and alpha_end for each iteration.
        verbose     prints information at the end of each epoch.
    """
    # Check parameters.
    assert nb_iter >= 1
    if nb_batches == None:
        nb_batches = max(1, len(samples) // 10)
    assert 1 <= nb_batches <= len(samples)
    assert alpha_end <= alpha_0
    
    # Initialize stuff.
    model = np.array(init, copy=True)
    denom = (alpha_0 / alpha_end)**(1./(nb_iter - 1))
    alpha = alpha_0

    for t in range(1, nb_iter):
        if verbose:
            print "Running epoch " + repr(t) + "..."
        # Shuffle the dataset.
        shuffledidxs = np.random.permutation(len(samples))
        # Split the dataset into nb_batches roughly equally sized batches.
        thresh = np.round(
            np.linspace(0, len(samples), num=nb_batches+1)
        ).astype(np.int32)
        # Keep track of average subgradient across the epoch for
        # info messages.
        avgsubgradnorm = 0

        for i in range(nb_batches):
            # Compute the corresponding batch.
            batch = [samples[idx] for idx 
                     in shuffledidxs[thresh[i]:thresh[i+1]]]
            # Compute the subgradient at the current point.
            subgrad = f_subgrad(nb_batches, model, batch)
            # Update the current model using the subgradient.
            model = model - alpha * subgrad
            avgsubgradnorm += np.linalg.norm(subgrad)
        # Update the learning rate.
        alpha = alpha / denom
        if verbose:
            print (
                "Average subgradient norm: " 
                + repr(avgsubgradnorm / nb_batches)
            )
    
    return model
