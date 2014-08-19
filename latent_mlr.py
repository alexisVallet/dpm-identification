""" Regularized latent multinomial logistic regression implementation.
"""
import numpy as np
import theano
import theano.tensor as T
from grid_search import GridSearchMixin

def compile_funcs():
    """ Compiles and returns theano functions. """

class BaseLatentMLR:
    def __init__(self, C, latent_function, latent_args, initbeta,
                 nb_coord_iter=1, nb_gd_iter=100, learning_rate=0.01,
                 verbose=False):
        # Basic parameters.
        self.C = C
        self.latent_function = latent_function
        self.latent_args = latent_args
        self.nb_coord_iter = nb_coord_iter
        self.nb_gd_iter = nb_gd_iter
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Theano shared variables and model information.
        self.nb_features, self.nb_classes = initbeta.shape
        self.beta = theano.shared(
            initbeta.astype(theano.config.floatX),
            name='beta',
            borrow=True
        )
        self.b = theano.shared(
            np.zeros((self.nb_classes,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        # Compile common theano functions.
        self.compile_funcs()

    def __getstate__(self):
        return {
            'C': self.C,
            'latfunc': self.latent_function,
            'latargs': self.latent_args,
            'nb_coord_iter': self.nb_coord_iter,
            'nb_gd_iter': self.nb_gd_iter,
            'learning_rate': self.learning_rate,
            'nb_features': self.nb_features,
            'nb_classes': self.nb_classes,
            'beta': self.beta.get_value(),
            'b': self.b.get_value()
        }

    def __setstate__(self, params):
        self.C = params['C']
        self.latent_function = params['latfunc']
        self.latent_args = params['latargs']
        self.nb_coord_iter = params['nb_coord_iter']
        self.nb_gd_iter = params['nb_gd_iter']
        self.learning_rate = params['learning_rate']
        self.nb_features = params['nb_features']
        self.nb_classes = params['nb_classes']
        self.beta = theano.shared(
            params['beta'],
            name='beta'
        )
        self.b = theano.shared(
            params['b'],
            name='b'
        )
        self.compile_funcs()

    def compile_funcs(self):
        # Prediction function.
        # Test_lat is a nb_classes * nb_samples * nb_features
        # tensor containing, for each class, the latent vectors for
        # each test sample. This format allows the use of theano's
        # batched_dot function to compute all the scores in one go.
        test_lat = T.tensor3('test_lat')
        predict_proba_sym = (
            T.nnet.sigmoid(T.batched_dot(test_lat, self.beta.T).T + self.b)
        )
        self._predict_proba = theano.function(
            [test_lat],
            predict_proba_sym
        )
        self._predict_label = theano.function(
            [test_lat],
            T.argmax(predict_proba_sym, axis=1)
        )

    def train(self, samples, labels):
        # Check parameters.
        assert labels.size == len(samples)
        for i in range(labels.size):
            assert labels[i] in range(self.nb_classes)
        
        nb_samples = len(samples)
        # Set and compile the theano gradient descent update function.
        lat_pos = theano.shared(
            np.empty([nb_samples, self.nb_features],
                     dtype=theano.config.floatX),
            name='lat_pos'
        )
        lat_neg = theano.shared(
            np.empty([nb_samples, self.nb_features],
                     dtype=theano.config.floatX),
            name='lat_neg'
        )
        # Cost function.
        regularization = (
            0.5 * (T.dot(T.flatten(self.beta), T.flatten(self.beta)) 
                   + T.dot(T.flatten(self.b), T.flatten(self.b)))
        )
        
        posdot = (T.dot(lat_pos, self.beta) + self.b)[
            T.arange(nb_samples), 
            labels
        ]
        losses = T.log(
            T.exp(posdot) /
            T.reshape(
                T.exp(T.dot(lat_neg, self.beta) + self.b).sum(axis=1,
                                                              keepdims=True),
                (nb_samples,))
        )
        cost = (
            regularization - self.C * T.sum(losses)
        )

        # Gradients.
        grad_beta = T.grad(cost, self.beta)
        grad_b = T.grad(cost, self.b)
        updates = [
            (self.beta, self.beta - self.learning_rate * grad_beta),
            (self.b, self.b - self.learning_rate * grad_b)
        ]
        # Theano function to compute cost and gradient norm while
        # updating the model at the same time.
        grad_descent = theano.function(
            inputs=[],
            outputs=[cost, 
                     T.sqrt(T.dot(T.flatten(grad_beta), T.flatten(grad_beta))
                            + T.dot(T.flatten(grad_b), T.flatten(grad_b)))],
            updates=updates            
        )
        new_lat_pos = np.empty(
            [nb_samples, self.nb_features],
            dtype=theano.config.floatX
        )
        new_lat_neg = np.empty(
            [nb_samples, self.nb_features],
            dtype=theano.config.floatX
        )
        bestmodel = np.empty(
            [self.nb_features, self.nb_classes],
            dtype=theano.config.floatX
        )
        bestcost = np.inf

        for t_coord in range(self.nb_coord_iter):
            # Compute the best positive latent vectors.
            new_lat_pos = self.latent_function(
                self.beta.get_value(),
                samples,
                labels,
                self.latent_args
            )
            lat_pos.set_value(new_lat_pos)
            
            for t_gd in range(self.nb_gd_iter):
                # Compute the best negative latent vectors.
                new_lat_neg = self.latent_function(
                    self.beta.get_value(),
                    samples,
                    labels,
                    self.latent_args
                )
                lat_neg.set_value(new_lat_neg)
                cost, gradnorm = grad_descent()

                if cost < bestcost:
                    bestmodel = (
                        np.array(self.beta.get_value(), copy=True),
                        np.array(self.b.get_value(), copy=True)
                    )
                    bestcost = cost
                
                if self.verbose:
                    print "Epoch " + repr(t_gd + 1)
                    print "Cost: " + repr(cost)
                    print "Gradient norm: " + repr(gradnorm)
        bestbeta, bestb = bestmodel
        self.b.set_value(bestb)
        self.beta.set_value(bestbeta)
        self.intercept_ = self.b.get_value()
        self.coef_ = self.beta.get_value()

    def predict_proba(self, samples):
        nb_samples = len(samples)
        beta_value = self.beta.get_value()
        nb_features, nb_classes = beta_value.shape
        test_latents = np.empty(
            [nb_classes, nb_samples, nb_features],
            dtype=theano.config.floatX
        )

        # Compile all latent values for each class in the test_latents
        # 3D tensor.
        for l in range(nb_classes):
            test_latents[l] = self.latent_function(
                beta_value,
                samples,
                np.repeat([l], nb_samples),
                self.latent_args
            )

        # Run the theano prediction function over it.
        return self._predict_proba(test_latents)

    def predict(self, samples):
        nb_samples = len(samples)
        beta_value = self.beta.get_value()
        nb_features, nb_classes = beta_value.shape
        test_latents = np.empty(
            [nb_classes, nb_samples, nb_features],
            dtype=theano.config.floatX
        )

        # Compile all latent values for each class in the test_latents
        # 3D tensor.
        for l in range(nb_classes):
            test_latents[l] = self.latent_function(
                beta_value,
                samples,
                np.repeat([l], nb_samples),
                self.latent_args
            )

        return self._predict_label(test_latents)

class LatentMLR(BaseLatentMLR, GridSearchMixin):
    pass

def _dummy_latent(beta, samples, labels, args):
    return np.vstack(samples)

class BaseMLR:
    """ Implementation of non-latent multinomial logistic regression based
        on latent MLR.
    """
    def __init__(self, C=0.1, nb_iter=100, learning_rate=0.001, verbose=False):
        self.C = C
        self.nb_iter = nb_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        
    def train(self, samples, labels):
        nb_features = samples[0].size
        nb_classes = np.unique(labels).size
        
        self.lmlr = LatentMLR(self.C, _dummy_latent, None,
                              np.zeros(
                                  [nb_features, nb_classes],
                                  dtype=theano.config.floatX
                              ), 
                              nb_coord_iter=1,
                              nb_gd_iter=self.nb_iter, 
                              learning_rate=self.learning_rate,
                              verbose=self.verbose)
        self.lmlr.train(samples, labels)
        self.intercept_ = self.lmlr.intercept_
        self.coef_ = self.lmlr.coef_

    def predict_proba(self, samples):        
        return self.lmlr.predict_proba(samples)

    def predict(self, samples):
        return self.lmlr.predict(samples)

class MLR(BaseMLR, GridSearchMixin):
    pass
