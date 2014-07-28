""" Generic binary latent logistic regression classifier.
"""
import numpy as np
import theano
import theano.tensor as T

_beta = T.vector('beta')
_b = T.scalar('b')
_phi = T.matrix('phi')
_prediction = theano.function(
    [_beta, _b, _phi],
    T.nnet.sigmoid(T.dot(_phi, _beta) + _b)
)

class LatentLR:
    def __init__(self, latent_function, latent_args=None, verbose=False):
        self.verbose = verbose
        self.latent_function = latent_function
        self.latent_args = latent_args
        
    def train(self, C, positives, negatives, initbeta, initbias=0,
              learning_rate=0.01, nb_coord_iter=1, nb_gd_epochs=100,
              eps=10E-5):
        assert learning_rate > 0
        nb_pos, nb_neg = (len(positives), len(negatives))
        nb_samples = nb_pos + nb_neg
        nb_features = initbeta.size

        # Theano variables.
        self.beta = theano.shared(initbeta.astype(theano.config.floatX))
        self.b = theano.shared(float(initbias))
        # Separate latent vector matrices for positives and negatives,
        # as they won't be modified as often.
        phi_pos = theano.shared(np.empty(
            [nb_pos, nb_features],
            dtype=theano.config.floatX
        ))
        phi_neg = theano.shared(np.empty(
            [nb_neg, nb_features],
            dtype=theano.config.floatX
        ))
        # Cost function.
        cost = 0.5 * (T.dot(self.beta, self.beta) + self.b**2) + C * (
            T.sum(T.nnet.softplus(T.dot(phi_neg, self.beta) + self.b))
            + T.sum(T.nnet.softplus(-T.dot(phi_pos, self.beta) + self.b))
        )

        # Update rules for gradient descent.
        grad_beta = T.grad(cost, self.beta)
        grad_b = T.grad(cost, self.b)
        updates = [
            (self.beta, self.beta - learning_rate * grad_beta),
            (self.b, self.b - learning_rate * grad_b)
        ]
        # Function to compute cost, gradients and update the model at the 
        # same time, while returning the gradient norm.
        grad_descent = theano.function(
            inputs=[],
            outputs=[cost, T.sqrt(T.dot(grad_beta, grad_beta) + grad_b**2)],
            updates=updates
        )
        
        # "Coordinate descent" of alternating optimization of positive
        # latents and model.
        new_phi_neg = np.empty(
            [nb_neg, nb_features],
            dtype=theano.config.floatX
        )
        new_phi_pos = np.empty(
            [nb_pos, nb_features],
            dtype=theano.config.floatX
        )
        bestmodel = None
        bestcost = np.inf
        for t_coord in range(nb_coord_iter):
            # Compute the best positive latent vectors.
            for i in range(nb_pos):
                new_phi_pos[i] = self.latent_function(
                    self.beta.get_value(),
                    positives[i],
                    self.latent_args
                )
            phi_pos.set_value(new_phi_pos)
            
            # Run gradient descent to optimize the cost function with
            # fixed positive latents.
            for t_gd in range(nb_gd_epochs):
                # Compute the best negative latent vectors.
                for i in range(nb_neg):
                    new_phi_neg[i] = self.latent_function(
                        self.beta.get_value(),
                        negatives[i],
                        self.latent_args
                    )
                phi_neg.set_value(new_phi_neg)
                cost, gradnorm = grad_descent()
                # Keep track of the best model found.
                if bestcost > cost:
                    bestmodel = (
                        np.array(self.beta.get_value(), copy=True),
                        self.b.get_value())
                    bestcost = cost

                if self.verbose:
                    print "Epoch " + repr(t_gd + 1)
                    print "Cost: " + repr(cost)
                    print "Gradient norm: " + repr(gradnorm)

                if gradnorm < eps:
                    break
        # Only keep the best model found.
        bestbeta, bestb = bestmodel
        self.b.set_value(bestb)
        self.beta.set_value(bestbeta)
        # Keeping the interface with scikit learn.
        self.intercept_ = self.b.get_value()
        self.coef_ = self.beta.get_value()
        
    def predict_proba(self, samples):
        nb_samples = len(samples)
        nb_features = self.beta.get_value().size
        phi = np.empty(
            [nb_samples, nb_features],
            dtype=theano.config.floatX
        )

        for i in range(nb_samples):
            phi[i] = self.latent_function(
                self.beta.get_value(),
                samples[i],
                self.latent_args
            )

        return _prediction(self.beta.get_value(), self.b.get_value(), phi)

def _dummy_latent(beta, sample, args):
    return sample

class BinaryLR:
    """ Binary non-latent logistic regression implemented in terms of 
        latent LR. For testing purposes mostly.
    """
    def __init__(self, verbose=False):
        self.llr = LatentLR(_dummy_latent, verbose=verbose)
    
    def fit(self, X, y, nb_iter=100, learning_rate=0.01, C=0.1):
        nb_samples, nb_features = X.shape
        positives = []
        negatives = []

        for i in range(nb_samples):
            if y[i] > 0:
                positives.append(X[i])
            else:
                negatives.append(X[i])
        self.llr.train(C, positives, negatives, np.zeros([nb_features]),
                       0, nb_coord_iter=1, nb_gd_epochs=nb_iter,
                       learning_rate=learning_rate)
        self.coef_ = self.llr.coef_
        self.intercept_ = self.llr.intercept_
    
    def predict_proba(self, X):
        nb_samples = X.shape[0]
        samples = []

        for i in range(nb_samples):
            samples.append(X[i])
        
        return self.llr.predict_proba(samples)
