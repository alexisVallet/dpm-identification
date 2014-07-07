""" Generic binary latent logistic regression classifier.
"""
import numpy as np
import multiprocessing as mp
from scipy.optimize import fmin_cg, fmin_l_bfgs_b, fmin_bfgs, fmin_ncg
import cPickle as pickle
import os

class BinaryLLR:
    def __init__(self, latent_function, C, verbose=False, 
                 algorithm='l-bfgs'):
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
            verbose            true if you want information messages 
                               printed at regular intervals.
            algorithm          algorithm to use for optimizing the the 
                               cost function. Must be one of 'cg' for 
                               conjugate gradient, 'bfgs' for BFGS, 
                               'l-bfgs' for L-BFGS or 'ncg' for the 
                               Newton-CG method. Note that L-BFGS is 
                               significantly more efficient in our 
                               implementation.
        """
        assert algorithm in ['cg', 'bfgs', 'l-bfgs', 'ncg']
        self.latent_function = latent_function
        self.C = C
        self.verbose = verbose
        self.algorithm = algorithm

    def cost_function_and_grad(self, negatives, poslatents, model):
        """ Compute function value and gradient in one go. About twice
            more efficient thant calling cost_function and cost_gradient
            separately.
        """
        nb_pos = poslatents.shape[0]
        nb_samples = nb_pos + len(negatives)
        nb_featuresp1 = model.size
        costinnersum = 0
        gradinnersum = np.zeros([nb_featuresp1])

        # Add up logistic loss for positive samples.
        for i in range(nb_pos):
            # Cost function.
            latvec = poslatents[i,:]
            modellatdot = np.vdot(model, latvec)
            costinnersum += np.log(1 + np.exp(-modellatdot))
            # Gradient.
            gradinnersum -= (
                (1.0 / (1 + np.exp(modellatdot)))
                * latvec
            )
        # Compute latent value and add up logistic loss for negative
        # samples.
        biaslessmodel = model[1:nb_featuresp1]
        for i in range(len(negatives)):
            # Cost function.
            latvec = self.latent_function(
                biaslessmodel,
                negatives[i]
            )
            biaslatvec = np.empty([nb_featuresp1])
            biaslatvec[0] = 1
            biaslatvec[1:nb_featuresp1] = latvec
            modellatdot = np.vdot(model, biaslatvec)
            costinnersum += np.log(1 + np.exp(modellatdot))
            # Gradient.
            gradinnersum += (
                (1.0 / (1 + np.exp(-modellatdot)))
                * biaslatvec
            )
        # Regularize and return. Ignore the bias for regularization.
        # Exclude bias from regularization.
        zerobiasmodel = np.empty([nb_featuresp1])
        zerobiasmodel[0] = 0
        zerobiasmodel[1:nb_featuresp1] = biaslessmodel
        cost = 0.5 * np.vdot(biaslessmodel, biaslessmodel) + self.C * costinnersum
        gradient = zerobiasmodel + self.C * gradinnersum

        if self.verbose:
            print "cost: " + repr(cost)
            print "model norm: " + repr(np.linalg.norm(model))
            print "gradient size: " + repr(gradient.size)
            print "gradient norm: " + repr(np.linalg.norm(gradient))
            print "gradient max: " + repr(gradient.max()) + " at " + repr(gradient.argmax())
            print "gradient min: " + repr(gradient.min())
            print "gradient avg: " + repr(gradient.mean())

        return (cost, gradient)

    def cost_function(self, negatives, poslatents, model):
        """ Computes the logistic cost function. Runs in
            time O(p + n * l) where l is the the runtime of
            the latent vector function, p is the number of
            positive samples and n is the number of negative samples.
        
        Arguments:
            negatives  negative samples to compute the cost function on.
            poslatents matrix of latent vectors as rows for positive
                       samples.
            model      model vector to compute the cost for.
        """
        nb_pos = poslatents.shape[0]
        nb_samples = nb_pos + len(negatives)
        nb_featuresp1 = model.size
        innersum = 0

        # Add up logistic loss for positive samples.
        for i in range(nb_pos):
            innersum += np.log(1 + np.exp(-np.vdot(model, poslatents[i,:])))
        # Compute latent value and add up logistic loss for negative
        # samples.
        biaslessmodel = model[1:nb_featuresp1]
        for i in range(len(negatives)):
            latvec = self.latent_function(
                biaslessmodel, 
                negatives[i]
            )
            modellatdot = (
                np.vdot(latvec, biaslessmodel) 
                + model[0] # add up the bias
            )
            innersum += np.log(1 + np.exp(modellatdot))
        
        # Regularize and return.
        # Exclude bias from regularization.
        cost = 0.5 * np.vdot(biaslessmodel,biaslessmodel) + self.C * innersum

        if self.verbose:
            print "cost: " + repr(cost)
        return cost

    def cost_gradient(self, negatives, poslatents, model):
        """ Computes the gradient of the cost function at a given
            point.

        Arguments:
            negatives  negative samples to compute the cost function on.
            poslatents matrix of latent vectors as rows for positive
                       samples.
            model      model vector to compute the cost for.
        """
        nb_pos = poslatents.shape[0]
        nb_samples = nb_pos + len(negatives)
        nb_featuresp1 = model.size
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
        for i in range(len(negatives)):
            latvec = self.latent_function(
                biaslessmodel,
                negatives[i]
            )
            # Add up the bias to the latent vector.
            biaslatvec = np.empty([nb_featuresp1])
            biaslatvec[0] = 1
            biaslatvec[1:nb_featuresp1] = latvec
            modellatdot = np.vdot(model, biaslatvec)
            innersum -= (
                (1.0 / (1 + np.exp(-modellatdot)))
                * biaslatvec
            )
        
        # Add up the gradient of the regularization term.
        # Exclude bias from regularization.
        zerobiasmodel = np.empty([nb_featuresp1])
        zerobiasmodel[0] = 0
        zerobiasmodel[1:nb_featuresp1] = biaslessmodel
        gradient = zerobiasmodel + self.C * innersum

        if self.verbose:
            print "model norm: " + repr(np.linalg.norm(model))
            print "gradient size: " + repr(gradient.size)
            print "gradient norm: " + repr(np.linalg.norm(gradient))
            print "gradient max: " + repr(gradient.max()) + " at " + repr(gradient.argmax())
            print "gradient min: " + repr(gradient.min())
            print "gradient avg: " + repr(gradient.mean())
        return gradient
        
    def fit(self, positives, negatives, initmodel, nbiter=4):
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
                    positives[i]
                )
                poslatents[i,1:nb_features+1] = latvec
                # bias
                poslatents[i,0] = 1
            if self.verbose:
                print "Optimizing the cost function for fixed positive latents..."
            # Optimizes the cost function for the fixed positive
            # latent vectors.
            warnflag = None
            if self.algorithm=='l-bfgs':
                currentmodel, fopt, d = (
                    fmin_l_bfgs_b(
                        lambda m: self.cost_function_and_grad(
                            negatives,
                            poslatents,
                            m
                        ),
                        currentmodel
                    )
                )
                warnflag = d['warnflag']
            elif self.algorithm=='cg':
                currentmodel, fopt, fcalls, gcalls, warnflag = (
                    fmin_cg(
                        lambda m: self.cost_function(
                            negatives,
                            poslatents,
                            m
                        ),
                        currentmodel,
                        lambda m: self.cost_gradient(
                            negatives,
                            poslatents,
                            m
                        ),
                        full_output=True
                    )
                )
            elif self.algorithm=='bfgs':
                currentmodel, fopt, gopt, Bopt, fcalls, gcalls, warnflag = (
                    fmin_bfgs(
                        lambda m: self.cost_function(
                            negatives,
                            poslatents,
                            m
                        ),
                        currentmodel,
                        lambda m: self.cost_gradient(
                            negatives,
                            poslatents,
                            m
                        ),
                        full_output=True
                    )
                )
            elif self.algorithm=='ncg':
                currentmodel, fopt, fcalls, gcalls, hcalls, warnflag = (
                    fmin_ncg(
                        lambda m: self.cost_function(
                            negatives,
                            poslatents,
                            m
                        ),
                        currentmodel,
                        lambda m: self.cost_gradient(
                            negatives,
                            poslatents,
                            m
                        ),
                        full_output = True
                    )
                )
            
            if self.verbose:
                if warnflag == 0:
                    print "Successfully converged."
                elif warnflag == 1:
                    print "Maximum number of iterations reached."
                else:
                    print "Gradient and/or cost were not changing."

        # Saves the results.
        self.model = currentmodel

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
            latvec = self.latent_function(biaslessmodel, samples[i])
            probas[i] = 1.0 / (1 + np.exp(
                -np.vdot(biaslessmodel, latvec)
                -self.model[0] # bias
            ))
        
        return probas
