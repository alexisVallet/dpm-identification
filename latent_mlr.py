""" Regularized latent multinomial logistic regression implementation.
"""
import numpy as np
import theano
import theano.tensor as T
from scipy.optimize import fmin_l_bfgs_b

from classifier import ClassifierMixin

class BaseLatentMLR:
    def __init__(self, C, latent_function, latent_args, initbeta,
                 nb_gd_iter=100, learning_rate=0.0001,
                 inc_rate=1.2, dec_rate=0.5, nb_samples=None, verbose=False):
        # Basic parameters.
        self.C = C
        self.latent_function = latent_function
        self.latent_args = latent_args
        self.nb_gd_iter = nb_gd_iter
        self.learning_rate = learning_rate
        self.dec_rate = dec_rate
        self.inc_rate = inc_rate
        self.nb_samples = nb_samples
        self.verbose = verbose

        # Theano shared variables and model information.
        self.nb_features, self.nb_classes = initbeta.shape
        self.beta = theano.shared(
            np.concatenate(
                (np.zeros((1, self.nb_classes), theano.config.floatX),
                 initbeta.astype(theano.config.floatX)),
                axis=0
            ),
            name='beta'
        )
        # Compile common theano functions.
        self.compile_funcs()

    def __getstate__(self):
        return {
            'C': self.C,
            'latfunc': self.latent_function,
            'latargs': self.latent_args,
            'nb_features': self.nb_features,
            'nb_classes': self.nb_classes,
            'beta': self.beta.get_value()
        }

    def __setstate__(self, params):
        self.C = params['C']
        self.latent_function = params['latfunc']
        self.latent_args = params['latargs']
        self.nb_features = params['nb_features']
        self.nb_classes = params['nb_classes']
        self.beta = theano.shared(
            params['beta'],
            name='beta',
        )
        self.compile_funcs()

    def compile_funcs(self):
        # Prediction function.
        # Test_lat is a nb_classes * nb_samples * nb_features
        # tensor containing, for each class, the latent vectors for
        # each test sample. This format allows the use of theano's
        # batched_dot function to compute all the scores in one go.
        test_lat = T.tensor3('test_lat')
        test_cst = T.matrix('test_cst')
        predict_proba_sym = (
            T.nnet.softmax(T.batched_dot(test_lat, self.beta[1:,:].T).T
                           + test_cst.T + self.beta[0,:])
        )
        self._predict_proba = theano.function(
            [test_lat, test_cst],
            predict_proba_sym
        )
        self._predict_label = theano.function(
            [test_lat, test_cst],
            T.argmax(predict_proba_sym, axis=1)
        )

    def train(self, samples, labels, valid_samples=[], valid_labels=None):
        # Check parameters.
        for i in range(labels.size):
            assert labels[i] in range(self.nb_classes)

        nb_samples = len(samples) if self.nb_samples == None else self.nb_samples
        # Set and compile the theano gradient descent update function.
        # Tensor for latent vectors.
        lat = theano.shared(
            np.empty([self.nb_classes, nb_samples, self.nb_features],
                     dtype=theano.config.floatX),
            name='lat'
        )
        # Tensor for latent constants.
        lat_cst = theano.shared(
            np.empty([self.nb_classes, nb_samples],
                     dtype=theano.config.floatX),
            name='lat_cst'
        )
        # Cost function.
        regularization = (
            0.5 * T.dot(T.flatten(self.beta), T.flatten(self.beta))
        )
        scores = T.batched_dot(lat, self.beta[1:,:].T).T + lat_cst.T + self.beta[0,:]
        losses = T.log(T.nnet.softmax(scores)[T.arange(nb_samples), labels])
        cost_sym = (
            regularization - self.C * T.sum(losses)
        )

        # Optimization of the cost with RPROP.
        weights_shape = self.beta.get_value().shape
        nb_weights = np.prod(weights_shape)
        # Keep step for each weight into a shared theano vector.
        steps = theano.shared(
            np.array(
                [self.learning_rate] * nb_weights,
                theano.config.floatX
            ).reshape(weights_shape),
            name='steps'
        )
        # Gradient of the cost function.
        grad = T.grad(cost_sym, self.beta)
        grad_f = theano.function(
            [],
            grad
        )
        # Perform the first iteration "manually", to initialize prev_grad properly.
        lat_val, cst_val = self.latent_function(
            self.beta.get_value()[1:,:],
            samples,
            labels,
            self.latent_args
        )
        assert lat_val.shape == (self.nb_classes, nb_samples, self.nb_features)
        assert cst_val.shape == (self.nb_classes, nb_samples)
        lat.set_value(lat_val)
        lat_cst.set_value(cst_val)
        init_grad = grad_f()
        self.beta.set_value(self.beta.get_value() - steps.get_value() 
                            * np.sign(init_grad))
        # Keep the gradient from the previous iteration into another shared theano
        # vector.
        prev_grad = theano.shared(
            init_grad,
            name='g(t-1)'
        )
        # Update rules for RPROP, scaling weight-wise step up when gradient signs 
        # agree, down when they disagree.
        # Vector containing 0 if the sign of the gradients disagree, 1 otherwise.
        sign_idx = T.iround((T.sgn(prev_grad * grad) + 1.)/2.).flatten()
        # Using the previously defined indices, we index into a matrix to get the 
        # new step vector.
        new_steps = T.stack(
            self.dec_rate * steps.flatten(), self.inc_rate * steps.flatten()
        )[sign_idx, T.arange(nb_weights)].reshape(weights_shape)
        # Specifies the updates at each iteration. We update the steps, the 
        # gradient from the previous iteration, and the actual descent.
        updates = [
            (steps, T.clip(new_steps, 10E-5, 0.1)),
            (prev_grad, grad),
            (self.beta, self.beta - steps * T.sgn(grad))
        ]
        # Full theano rprop function. Outputs the cost and gradient norm at the 
        # current point, and updates all the variables accordingly.
        rprop_descent = theano.function(
            [],
            [cost_sym, grad.norm(2)],
            updates=updates
        )
        eps = 10E-3
        
        for t_gd in range(self.nb_gd_iter):
            # Compute the best negative latent vectors.
            lat_val, cst_val = self.latent_function(
                self.beta.get_value()[1:,:],
                samples,
                labels,
                self.latent_args
            )
            lat.set_value(lat_val)
            lat_cst.set_value(cst_val)
            cost_val, grad_norm = rprop_descent()
            if self.verbose:
                print "Epoch " + repr(t_gd + 1)
                print "Cost: " + repr(cost_val)
                print "Gradient norm: " + repr(grad_norm)
                print "Mean step size: " + repr(steps.get_value().mean())
            if grad_norm <= eps:
                break
        self.intercept_ = self.beta.get_value()[0,:]
        self.coef_ = self.beta.get_value()[1:,:]
        # Trick for Theano to free GPU memory, hopefully.
        lat.set_value([[[]]])
        lat_cst.set_value([[]])

    def predict_proba(self, samples):
        nb_samples = len(samples)
        beta_value = self.beta.get_value()
        nb_featuresp1, nb_classes = beta_value.shape
        test_lat, test_cst = self.latent_function(
            beta_value[1:,:],
            samples,
            np.repeat([0], nb_samples),
            self.latent_args
        )

        # Run the theano prediction function over it.
        return self._predict_proba(test_lat, test_cst)

    def predict(self, samples):
        nb_samples = len(samples)
        beta_value = self.beta.get_value()
        nb_featuresp1, nb_classes = beta_value.shape
        test_lat, test_cst = self.latent_function(
            beta_value[1:,:],
            samples,
            np.repeat([0], nb_samples),
            self.latent_args
        )            

        return self._predict_label(test_lat, test_cst)

class LatentMLR(BaseLatentMLR, ClassifierMixin):
    pass

def _dummy_latent(beta, samples, labels, args):
    nb_features, nb_classes = beta.shape
    nb_samples = len(samples)
    latents = np.empty(
        [nb_classes, nb_samples, nb_features],
        dtype=theano.config.floatX
    )
    stacked = np.vstack(samples)
    for i in range(nb_classes):
        latents[i] = stacked
    return (latents, np.zeros([nb_classes, nb_samples], theano.config.floatX))

class BaseMLR:
    """ Implementation of non-latent multinomial logistic regression based
        on latent MLR.
    """
    def __init__(self, C=0.1, nb_iter=100, learning_rate=0.001, 
                 inc_rate=1.2, dec_rate=0.5, verbose=False):
        self.C = C
        self.nb_iter = nb_iter
        self.learning_rate = learning_rate
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.verbose = verbose
        
    def train(self, samples, labels, valid_samples=[], valid_labels=None):
        nb_features = samples[0].size
        nb_classes = np.unique(labels).size
        
        self.lmlr = LatentMLR(self.C, _dummy_latent, None,
                              np.zeros(
                                  [nb_features, nb_classes],
                                  dtype=theano.config.floatX
                              ),
                              nb_gd_iter=self.nb_iter,
                              learning_rate=self.learning_rate,
                              inc_rate=self.inc_rate,
                              dec_rate=self.dec_rate,
                              verbose=self.verbose)
        self.lmlr.train(samples, labels, valid_samples, valid_labels)
        self.intercept_ = self.lmlr.intercept_
        self.coef_ = self.lmlr.coef_

    def predict_proba(self, samples):        
        return self.lmlr.predict_proba(samples)

    def predict(self, samples):
        return self.lmlr.predict(samples)

class MLR(BaseMLR, ClassifierMixin):
    pass
