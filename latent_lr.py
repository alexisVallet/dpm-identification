""" Generic binary latent logistic regression classifier.
"""
import numpy as np
from scipy.optimize import 

class BinaryLLR:
    def __init__(self, latent_function, C, verbose=False):
        """ Initializes the model with a specific function for latent
            computation.

        Arguments:
            latent_function    function returning taking as argument
                               a model vector beta and a sample x, 
                               returning the best possible latent vector
                               phi(x, z) for all possible latent values
                               z. Formally:
                  latent_function(beta, x) = argmax_z (beta . phi(x, z))
                               Where beta is a nb_feature dimensional
                               model vector, and x is a nb_feature 
                               dimensional training sample. Should return 
                               an nb_feature dimensional latent vector.
            C                  soft margin parameter.
        """
        self.latent_function = latent_function
        self.C = C
        self.verbose = verbose

    def cost_function(self, X, poslatents, model):
        """ Computes the logistic cost function. Runs in
            time O(p + n * l) where l is the the runtime of
            the latent vector function, p is the number of
            positive samples and n is the number of negative samples.
        
        Arguments:
            X          samples to compute the cost function on,
                       ordered with positive samples at the beginning.
            poslatents matrix of latent vectors as rows for positive
                      samples.
            model     model vector to compute the cost for.
        """
        nb_samples, nb_featuresp1 = X.shape
        nb_pos = poslatents.shape[0]
        innersum = 0

        # Add up logistic loss for positive samples.
        for i in range(nb_pos):
            innersum += np.log(1 + np.exp(-np.vdot(model, X[i,:])))
        # Compute latent value and add up logistic loss for negative
        # samples.
        biaslessmodel = model[1:nb_featuresp1]
        for i in range(nb_pos, nb_samples):
            latvec = self.latent_function(
                biaslessmodel, 
                X[i,1:nb_features]
            )
            modellatdot = (
                np.vdot(latvec, biaslessmodel) 
                + model[nb_featuresp1 - 1] # add up the bias
            )
            innersum += np.log(1 + np.exp(modellatdot))
        
        # Regularize and return.
        return 0.5 * np.vdot(model,model) + self.C * innersum

    def cost_gradient(self, X, poslatents, model):
        """ Computes the gradient of the cost function at a given
            point.

        Arguments:
            X          samples to compute the cost function on,
                       ordered with positive samples at the beginning.
            poslatents matrix of latent vectors as rows for positive
                      samples.
            model     model vector to compute the cost for.
        """
        nb_samples, nb_featuresp1 = X.shape
        nb_pos = poslatents.shape[0]
        innersum = np.zeros([nb_featuresp1])

        # Compute gradient for positive samples.
        for i in range(nb_pos):
            latvec = poslatents[i,:]
            innersum += (
                (1.0 / (1 + np.exp(np.vdot(model, latvec))))
                * latvec
            )
        # Compute latent vector and gradient for negative samples.
        biaslessmodel = model[1:nb_featuresp1]
        for i in range(nb_pos, nb_samples):
            latvec = self.latent_function(
                biaslessmodel,
                X[i,1:nb_featuresp1]
            )
            # Add up the bias to the latent vector.
            biaslatvec = np.empty([nb_featuresp1])
            biaslatvec[0] = 1
            biaslatvec[1:nb_featuresp1] = latvec
            modellatdot = np.vdot(model, biaslatvec)
            innersum += (
                (1.0 / (1 + np.exp(-modellatdot)))
                * biaslatvec
            )
        
        # Add up the gradient of the regularization term.
        return model + self.C + innersum
        
    def fit(self, X, y, initmodel, nbiter):
        """ Fits the model against data X with class labels y.

        Arguments:
            X         nb_samples by nb_features matrix of sample vectors.
            y         nb_samples vector of class labels in the {0, 1} set.
            initmodel an initial nb_features dimensional vector to 
                      optimize.
            nbiter    number of iterations of coordinate descent to run.
        """
        # Checks that everything is in order.
        nb_samples, nb_features = X.shape
        assert initmodel.size = nb_features + 1
        assert y.size == nb_samples
        assert (y == 0 | y == 1).all()
        assert self.latent_function != None
        
        # sort the data so positive samples are at the beginning.
        nb_pos = (y == 1).sum()
        X_ = np.empty_like(X)
        y_ = np.empty_like(y)

        posidx = 0
        negidx = nb_pos
        for i in range(nb_samples):
            if y[i] == 1:
                X_[posidx,:] = X[i,:]
                y_[posidx] = 1
                posidx += 1
            else:
                X_[negidx,:] = X[i,:]
                y_[negidx] = 0
                negidx += 1
        
        currentmodel = np.array([nb_features + 1])
        # bias
        currentmodel[0] = 0
        currentmodel[1:nb_features+1] = initmodel

        # Iteratively optimize the cost function using the coordinate
        # descent approach.
        for t in range(nbiter):
            if self.verbose:
                print "Running iteration " + repr(nbiter)
                print "Computing latent vectors for positive samples..."
            # Compute latent vectors for positive samples.
            poslatents = np.empty([nb_pos, nb_features + 1])
            
            for i in range(nb_pos):
                latvec = latent_function(
                    currentmodel[1:nb_features],
                    X_[i]
                )
                poslatents[i,1:nb_features+1] = latvec
                # bias
                poslatents[0] = 1

            if self.verbose:
                print "Optimizing the cost function for fixed positive latents..."
            # Optimizes the cost function for the fixed positive
            # latent vectors.
            currentmodel, fopt, func_calls, grad_calls, warnflag = (
                scipy.optimize.fmin_cg(
                    lambda m: self.cost_function(
                        X_,
                        poslatents,
                        m
                    ),
                    currentmodel,
                    fprime=lambda m: self.cost_gradient(
                        X_,
                        poslatents,
                        m
                    ),
                    full_output=True
                )
            )
            if self.verbose:
                print "Optimization resulte:"
                print "min value: " + repr(fopt)
                print "cost calls: " + repr(func_calls)
                print "gradient calls: " + repr(grad_calls)
                if warnflag == 0:
                    print "Successfully converged."
                elif warnflag == 1:
                    print "Maximum number of iterations reached."
                else:
                    print "Gradient and/or cost were not changing."

        # Saves the results.
        self.model = currentmodel
