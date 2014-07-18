""" Generic binary latent logistic regression classifier.
"""
import numpy as np
import multiprocessing as mp
from scipy.optimize import fmin_cg, fmin_l_bfgs_b, fmin_bfgs, fmin_ncg
import cPickle as pickle
import os
import theano
import theano.tensor as T
from theano.tensor.nnet import softplus

from ssm import ssm

class BinaryLLR:
    def __init__(self, latent_function, C, latent_args=None, 
                 verbose=False):
        """ Initializes the model with a specific function for latent
            computation.

        Arguments:
            latent_function    function returning taking as argument
                               a model vector beta and a sample x, 
                               returning the best possible latent vector
                               phi(x, z) for all possible latent values
                               z. Formally:
                  latent_function(beta, x) = argmax_z (beta . phi(x, z))
                               Where beta is an arbitrary dimensional 
                               model vector, and x is a nb_features 
                               dimensional vector. Should return an 
                               nb_feature dimensional latent vector. 
                               While there is no strict requirement, the
                               training procedure performs best if the
                               function is a maximum of functions linear
                               in beta. Other convex functions may also
                               perform well.
            C                  soft margin parameter.
            latent_args        additional arguments to (optionally) pass
                               to the latent_function.
            verbose            true if you want information messages 
                               printed at regular intervals.
        """
        self.latent_function = latent_function
        self.latent_args = latent_args
        self.C = C
        self.verbose = verbose
        # Symbolic theano expressions for the cost function and its 
        # subgradient with fixed latent vectors. The latent vectors 
        # themselves will be of recomputed in the case of negative samples 
        # as a preprocessing step.
        # Beta is the model vector.
        self.beta = T.vector('beta')
        # Phi is the matrix of precomputed latent vectors for beta as rows.
        self.phi = T.matrix('phi')
        # y is the vector of class labels.
        self.y = T.vector('y', dtype='int32')
        # Single element log loss and gradient.
        self.latent = T.vector('phi_i')
        self.label = T.scalar('yi', dtype='int32')
        single_log_loss = T.log(1 + T.exp(-self.label*T.dot(self.latent, self.beta)))
        self.log_loss_f = theano.function(
            [self.beta, self.latent, self.label],
            single_log_loss
        )
        log_loss_grad = T.grad(single_log_loss, self.beta)
        self.log_loss_grad_f = theano.function(
            [self.beta, self.latent, self.label],
            log_loss_grad
        )
        # Elementwise log loss, taking advantage of theano's softplus.
        self.log_loss = softplus(T.dot(-T.diag(self.y), T.dot(self.phi, self.beta)))
        # beta0 is the model beta with bias coefficient zeroed out.
        beta0 = T.set_subtensor(self.beta[0], 0)
        # Cost function. In this case, phi should contain all the latent 
        # vectors, and y all the corresponding labels.
        regularization = 0.5 * T.dot(beta0, beta0)
        innersum = self.C * T.sum(self.log_loss)
        self.cost_sym = (
            regularization + innersum
        )
        self.cost = theano.function(
            [self.beta, self.phi, self.y], 
            self.cost_sym
        )
        if verbose:
            print "Cost function: " + theano.pp(self.cost_sym)
        # Stochastic subgradient of the cost function. m will be the 
        # number of mini-batches. In this case, phi should hold only the 
        # latent vectors of the samples in the mini-batch, and y only the 
        # corresponding labels.
        self.m = T.scalar('m', dtype='int32')
        self.cost_subgrad_sym = (
            T.grad(regularization, self.beta) + self.m * T.grad(innersum, self.beta)
        )
        self.cost_subgrad = theano.function(
            [self.beta, self.phi, self.y, self.m],
            self.cost_subgrad_sym
        )
        if verbose:
            print (
                "Stochastic subgradient: " 
                + theano.pp(self.cost_subgrad_sym)
            )

    def loss_function(self, negatives, poslatents, model, sample):
        """ Computes the loss function on a single sample.

        Arguments:
            negatives
                list of negative samples.
            poslatents
                list of precomputed latent vectors for positive samples.
            model
                model to compute the loss against.
            sample
                (i, yi) tuple where i is the index of the sample in either
                negatives or poslatents, and yi is the label (-1 for negative,
                +1 for positive).
        Returns:
            the logarithmic loss of the sample's latent vector against the model.
        """
        idx, label = sample
        latvec = None
        if label > 0:
            latvec = poslatents[idx]
        else:
            latvec = np.empty([model.size])
            latvec[1:] = self.latent_function(
                model[1:],
                negatives[idx],
                self.latent_args
            )
            latvec[0] = 1
        
        return self.log_loss_f(model, latvec, label)

    def loss_subgrad(self, negatives, poslatents, model, sample):
        idx, label = sample
        latvec = None
        if label > 0:
            latvec = poslatents[idx]
        else:
            latvec = np.empty([model.size])
            latvec[1:] = self.latent_function(
                model[1:],
                negatives[idx],
                self.latent_args
            )
            latvec[0] = 1
        
        return self.log_loss_grad_f(model, latvec, label)

    def cost_function(self, negatives, poslatents, model):
        """ Computes the logistic cost function.
        
        Arguments:
            negatives  negative samples to compute the cost function on.
            poslatents matrix of latent vectors as rows for positive
                       samples.
            model      model vector to compute the cost for.
        """
        # Put latent vectors in a data structure suitable for Theano.
        nb_pos, nb_featuresp1 = poslatents.shape[0:2]
        latents = np.empty([nb_pos + len(negatives), nb_featuresp1])
        latents[0:nb_pos] = poslatents
        biaslessmodel = model[1:]
        for i in range(len(negatives)):
            # Compute the latent vector.
            latents[nb_pos+i,1:] = self.latent_function(
                biaslessmodel,
                negatives[i],
                self.latent_args
            )
        # Set biases for negative latent vectors.
        latents[nb_pos:,0] = 1
        # Set up labels.
        labels = np.empty([nb_pos + len(negatives)], dtype=np.int32)
        labels[0:nb_pos] = 1
        labels[nb_pos:] = -1

        return self.cost(model, latents, labels)

    def cost_stochastic_subgradient(self, poslatents, negatives,
                                    nb_batches, model, labelledsamples):
        """ Compute a subgradient of the cost function 
            on randomly selected samples at a given point.

        Arguments:
            model    model vector corresponding to the point at
                     which we want to evaluate the cost subgradient.
            nb_batches
                     the number of batches.
            labelledsamples
                     array-like of (sampleidx, label) pairs where
                     sampleidx is the index of the sample in the positives
                     or negatives list, and the labels should be 1 for 
                     positive samples, -1 for negative samples.
        Returns:
            a valid subgradient of the cost function at the given point.
        """
        # Set up data structure for Theano.
        nb_featuresp1 = model.size
        nb_samples = len(labelledsamples)
        latents = np.empty([nb_samples, nb_featuresp1])
        labels = np.empty([nb_samples], dtype=np.int32)
        biaslessmodel = model[1:]
        
        for i in range(nb_samples):
            (idx, label) = labelledsamples[i]
            if label > 0:
                latents[i] = poslatents[idx]
                labels[i] = 1
            else:
                latents[i,1:] = self.latent_function(
                    biaslessmodel,
                    negatives[idx],
                    self.latent_args
                )
                latents[i,0] = 1
                labels[i] = -1
        
        return self.cost_subgrad(model, latents, labels, 1)

    def fit(self, positives, negatives, initmodel, nbiter=4, nb_opt_iter=50):
        """ Fits the model against positive and negative samples
            given an initial model to optimize. It should be noted
            that positives and negative samples may be any python
            datatype, as their handling is defined by the latent function.

        Arguments:
            positives array of positive samples.
            negatives array of negative samples.
            initmodel an initial nb_features dimensional model vector to 
                      optimize.
            nbiter    number of iterations of coordinate descent to run.
        """
        # Checks that everything is in order.
        nb_samples = len(positives) + len(negatives)
        nb_features = initmodel.size
        assert self.latent_function != None
        currentmodel = np.empty([nb_features + 1])
        # bias
        currentmodel[0] = 0
        currentmodel[1:nb_features+1] = initmodel
        # Keep track of the best model encountered.
        # As the "coordinate descent" approach is mostly heuristic,
        # it is not guaranteed to converge to the minimum.
        bestmodel = None
        bestcost = np.inf

        # Iteratively optimize the cost function using the coordinate
        # descent approach.
        for t in range(nbiter):
            if self.verbose:
                print "Running iteration " + repr(t)
                print "Computing latent vectors for positive samples..."
            # Compute latent vectors for positive samples.
            poslatents = np.empty([len(positives), nb_features + 1])
            
            for i in range(len(positives)):
                latvec = self.latent_function(
                    currentmodel[1:nb_features+1],
                    positives[i],
                    self.latent_args
                )
                poslatents[i,1:nb_features+1] = latvec
                # bias
                poslatents[i,0] = 1
            if self.verbose:
                print "Optimizing the cost function for fixed positive latents..."
            # Optimizes the cost function for the fixed positive
            # latent vectors.
            # Associate class labels to samples as the subgradient
            # requires it. Simply pass around the indexes, which we
            # will need to refer to positive latents anyway.
            labelledsamples = (
                zip(range(len(positives)), [1] * len(positives)) +
                zip(range(len(negatives)), [-1] * len(negatives))
            )
            currentmodel = ssm(
                currentmodel,
                labelledsamples,
                lambda nb,m,s: self.cost_stochastic_subgradient(
                    poslatents,
                    negatives,
                    nb,
                    m,
                    s
                ),
                f=lambda m: self.cost_function(
                    negatives,
                    poslatents,
                    m
                ),
                alpha_0=0.01,
                learning_rate='constant',
                verbose=self.verbose,
                nb_iter=nb_opt_iter
            )
            # Keep track of the best encountered model.
            currentcost = self.cost_function(
                negatives,
                poslatents,
                currentmodel
            )
            if currentcost < bestcost:
                bestmodel = np.array(currentmodel, copy=True)
                bestcost = currentcost

        # Saves the results.
        self.model = bestmodel

    def predict_proba(self, samples):
        """ Returns probabilities of each sample being a positive sample.
        
        Arguments:
            samples    array of samples to predict probabilities for. Each
                       sample should be of the datatype accepted by 
                       self.latent_function .
        Returns:
           nb_sample numpy vector of probabilities.
        """
        nb_sample = len(samples)
        probas = np.empty([nb_sample])

        biaslessmodel = self.model[1:]
        for i in range(nb_sample):
            latvec = self.latent_function(biaslessmodel, samples[i], self.latent_args)
            probas[i] = 1.0 / (1 + np.exp(
                -np.vdot(biaslessmodel, latvec)
                -self.model[0] # bias
            ))
        
        return probas

def _dummy_latent_function(model, sample, args):
    return sample

class BinaryLR:
    """ Binary non-latent logistic regression implemented in terms of latent
        LR. For testing purposes mostly.
    """
    def __init__(self, C, verbose=False):
        self.llr = BinaryLLR(_dummy_latent_function, C, verbose=verbose)
    
    def fit(self, X, y):
        nb_samples, nb_features = X.shape
        positives = []
        negatives = []

        for i in range(nb_samples):
            if y[i] > 0:
                positives.append(X[i])
            else:
                negatives.append(X[i])
        self.llr.fit(positives, negatives, np.zeros([nb_features]), 1,
                     nb_opt_iter=100)
        self.coef_ = self.llr.model[1:]
    
    def predict_proba(self, X):
        nb_samples = X.shape[0]
        samples = []

        for i in range(nb_samples):
            samples.append(X[i])
        
        return self.llr.predict_proba(samples)
