""" Pegasos algorithm adapted for latent logistic regression. See 
    (Shai, Yorann, 2007) for more details about the algorithm.
"""
import numpy as np
import theano as th

def cost_function(loss, C, samples, model):
    innersum = 0

    for sample in samples:
        innersum += loss(model, sample)

    return 0.5 * np.vdot(model,model) + C * innersum

def subgradient_function(loss_subgrad, C, samples, model):
    innersum = 0

    for sample in samples:
        innersum += loss_subgrad(model, sample)

    return model + C * innersum

def pegasos(loss_subgrad, samples, C, initmodel, nb_iter=None, 
            loss=None, eps=10E-7, verbose=False):
    """ Minimizes a cost function expressed as a regularized sum of
        of a loss function using the Pegasos algorithm. The function
        minimized is:
        f(model) = 
            0.5||model||^2 + C * sum_i loss(model, samples[i])

    Arguments:
            where model is model vector, xi is a training sample and
            yi its binary class label in {-1; 1}. While model needs
            to be a vector, xi can be any element.
        loss_subgrad
            loss subgradient for a single sample, e.g. for logistic
            regression:
                loss_subgrad(model, (xi, yi)) = 
                  -yi * xi / (1 + exp(yi * xi . model))
        samples
            list of training samples to optimize the function for.
        labels
            list of labels associated to each training sample.
        C
            regularization parameter. Should be positive.
        initmodel
            initial model vector to start with. In many cases, this
            can be the 0 vector.
        nb_iter
            number of iterations to run. Note that the algorithm is
            proven to converge with high probability to a solution
            of accuracy eps in O(1/eps), and the number of iterations
            scales linearly with C. If None, a sensible number will be
            chosen depending on C and for a 10E-5 accuracy.
        loss
            loss function for a single sample, e.g. for logistic
            regression:
                loss(model, (xi, yi)) = log(1 + exp(-yi * xi . model))
            will be called once in a while (every 10 times we go across
            the dataset size).
    Returns:
        a new model vector which minimizes the cost function.
    """
    # Check parameters.
    assert isinstance(C, float)
    assert C > 0
    if nb_iter == None:
        assert C > eps
        nb_iter = int(round(C / eps))
    
    model = np.array(initmodel, copy=True)
    # Keep track of best value found overall, as algorithm tends to
    # overshoot.
    bestcost = np.inf
    bestmodel = None

    for t in range(1, nb_iter + 1):
        # At each iteration, pick a random sample.
        sample = samples[np.random.randint(len(samples))]
        # Update the step size.
        step = C / t
        # Compute the subgradient, and update the model accordingly.
        subgrad = model + C * len(samples) * loss_subgrad(model, sample)
        model -= step * subgrad
        if t % (10 * len(samples)) == 0:
            # Print some information every time we iterate through
            # 10 sample sizes.
            fullcost = cost_function(loss, C, samples, model)
            if fullcost < bestcost:
                bestcost = fullcost
                bestmodel = np.array(model, copy=True)
            if verbose:
                fullsubgrad = subgradient_function(loss_subgrad, C, samples, model)
                print "Iteration " + repr(t)
                print "Cost: " + repr(fullcost)
                print "Subgradient norm: " + repr(np.linalg.norm(fullsubgrad))
                print "Step: " + repr(step)
    
    if bestmodel == None:
        return model
    else:
        return bestmodel
