""" Simple implementation of logistic regression to test
    gradient descent minimization algorithms.
"""
import numpy as np
import theano
import theano.tensor as T

from pegasos import pegasos

class LogisticRegression:
    def __init__(self, C, verbose=False):
        # Basic attributes.
        self.C = C
        self.verbose = verbose
        # Theano functions for the loss function
        # and its subgradient, to be fed to gradient descent
        # algorithms.
        beta = T.vector('beta')
        xi = T.vector('xi')
        yi = T.scalar('yi', dtype='int32')
        log_loss_sym = T.log(1 + T.exp(-yi * T.dot(xi, beta)))
        self.log_loss = theano.function(
            [beta, xi, yi],
            log_loss_sym
        )
        log_loss_subgrad_sym = T.grad(
            log_loss_sym,
            beta
        )
        self.log_loss_subgrad = theano.function(
            [beta, xi, yi],
            log_loss_subgrad_sym
        )
        # Theano function for probability prediction, which is
        # just an elementwise sigmoid function.
        X = T.matrix('X')
        self.sigmoid = theano.function(
            [beta, X],
            T.nnet.sigmoid(T.dot(X, beta))
        )

    def loss(model, sample):
        xi, yi = sample

        return self.log_loss(model, xi, yi)

    def loss_subgrad(model, sample):
        xi, yi = sample

        return self.log_loss_subgrad(model, xi, yi)
    
    def fit(self, X, y):
        nb_samples, nb_features = X.shape
        # Add bias parameter.
        X_ = np.empty([nb_samples, nb_features + 1])
        X_[:,0] = 1
        X_[1:] = X
        # Convert to right data structure for pegasos.
        samples = map(lambda i: (X_[i], y[i]), range(nb_samples))
        # Optimize model with pegasos.
        self.model = pegasos(
            self.loss_subgrad,
            samples,
            self.C,
            np.zeros([nb_features + 1])
        )
        # Save up the model in the same format as scikit learn.
        self.coef_ = self.model[1:]

    def predict_proba(self, X):
        # Add up the bias.
        nb_samples, nb_features = X.shape
        X_ = np.empty([nb_samples, nb_features + 1])
        X_[:,0] = 1
        X_[1:] = X

        return self.sigmoid(self.model, X_)
