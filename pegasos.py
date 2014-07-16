""" Pegasos algorithm adapted for latent logistic regression. See 
    (Shai, Yorann, 2007) for more details about the algorithm.
"""
import numpy as np
import theano as th

def pegasos(loss_subgrad, samples, C, initmodel, nb_iter=None, 
            loss=None, verbose=False):
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
            will NOT be called by algorithm, unless verbose is set to
            True in which case it will be used to compute the full cost
            function to be displayed on a regular basis.
    Returns:
        a new model vector which minimizes the cost function.
    """
    # Check parameters.
    assert C > 0
    if nb_iter == None:
        assert C > 10E-5
        nb_iter = int(round(C / 10E-5))
    
    model = np.array(initmodel, copy=True)

    for t in range(1, nb_iter + 1):
        sample = np.random.choice(samples)
        step = C / t
        subgrad = model + C 
        model -= step * subgrad
