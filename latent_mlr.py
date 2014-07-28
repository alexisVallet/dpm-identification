""" Regularized latent multinomial logistic regression implementation.
"""
import numpy as np
import theano
import theano.tensor as T

class LatentMLR:
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
            initbeta.astype(theano.config.floatX)
        )
        self.b = theano.shared(
            np.zeros(self.nb_classes, dtype=theano.config.floatX)
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
        )
        self.b = theano.shared(
            params['b'],
        )
        self.compile_funcs()

    def compile_funcs(self):
        # Prediction function.
        test_lat = T.matrix('test_lat')
        predict_proba_sym = (
            T.nnet.sigmoid(T.dot(test_lat, self.beta) + self.b)
        )
        self._predict_proba = theano.function(
            [test_lat],
            predict_proba_sym
        )
        self._predict_label = theano.function(
            [test_lat],
            T.argmax(predict_proba_sym, axis=1)
        )

    def fit(self, samples, labels):
        # Check parameters.
        assert labels.size == len(samples)
        for i in range(labels.size):
            assert labels[i] in range(self.nb_classes)
        
        nb_samples = len(samples)
        # Set and compile the theano gradient descent update function.
        lat_pos = theano.shared(
            np.empty([nb_samples, self.nb_features],
                     dtype=theano.config.floatX)
        )
        lat_neg = theano.shared(
            np.empty([nb_samples, self.nb_features],
                     dtype=theano.config.floatX)
        )
        flat = lambda t: T.reshape(t, [T.prod(t.shape)])
        # Cost function.
        sample_betas = self.beta[T.arange(nb_samples), labels].T
        cost = (
            0.5 * (T.dot(flat(self.beta), flat(self.beta)) 
                   + T.dot(self.b, self.b))
            - self.C * T.sum(
                T.log(
                    T.exp(T.batched_dot(lat_pos, sample_betas) + self.b) /
                    T.sum(T.exp(T.dot(lat_neg, self.beta) + self.b), axis=1)
                )
            )
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
                     T.sqrt(T.dot(flat(grad_beta), flat(grad_beta))
                            + T.dot(flat(grad_b), flat(grad_b)))],
            updates=updates            
        )
        new_lat_pos = None
        new_lat_neg = None
        bestmodel = None
        bestcost = np.inf

        for t_coord in range(self.nb_coord_iter):
            # Compute the best positive latent vectors.
            new_lat_pos = self.latent_function(
                self.beta.get_value(),
                samples,
                self.latent_args
            )
            lat_pos.set_value(new_lat_pos)
            
            for t_gd in range(self.nb_gd_iter):
                # Compute the best negative latent vectors.
                new_lat_neg = self.latent_function(
                    self.beta.get_value(),
                    samples,
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

                if gradnorm < eps:
                    break
        bestbeta, bestb = bestmodel
        self.b.set_value(bestb)
        self.beta.set_value(bestbeta)
        self.intercept_ = self.b.get_value()
        self.coef_ = self.beta.get_value()

    def predict_proba(self, samples):
        latents = self.latent_function(
            self.beta.get_value(),
            samples,
            self.latent_args
        )

        return self._predict_proba(latents)

    def predict(self, samples):
        latents = self.latent_function(
            self.beta.get_value(),
            samples,
            self.latent_args
        )

        return self._predict_label(latents)

def _dummy_latent(beta, samples, args):
    return np.vstack(samples)

class MLR:
    """ Implementation of non-latent multinomial logistic regression based
        on latent MLR, for testing purposes.
    """
    def __init__(self, C, nb_iter=100, learning_rate=0.01, verbose=False):
        self.C = C
        self.nb_iter = nb_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        
    def fit(self, X, y):
        nb_samples, nb_features = X.shape
        nb_classes = np.unique(y).size
        
        self.lmlr = LatentMLR(self.C, _dummy_latent, None,
                              np.zeros([nb_features, nb_classes]), 
                              nb_coord_iter=1,
                              nb_gd_iter=self.nb_iter, 
                              learning_rate=self.learning_rate,
                              verbose=self.verbose)
        samples = []

        for i in range(nb_samples):
            samples.append(X[i])
        self.lmlr.fit(samples, y)
        self.intercept_ = self.lmlr.intercept_
        self.coef_ = self.lmlr.coef_

    def predict_proba(self, X):
        samples = []

        for i in range(X.shape[0]):
            samples.append(X[i])
        
        return self.lmlr.predict_proba(self, samples)
