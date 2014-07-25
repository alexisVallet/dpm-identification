""" Theano based implementation of logistic regression using mini-batch
    stochastic gradient descent. For testing purposes.
"""
import numpy as np
import theano 
import theano.tensor as T

def _compile_func():
    beta = T.vector('beta')
    b = T.scalar('b')
    X = T.matrix('X')
    y = T.vector('y')
    C = T.scalar('C')
    params = [beta, b, X, y, C]
    cost = 0.5 * (T.dot(beta, beta) + b * b) + C * T.sum(
        T.nnet.softplus(
            -T.dot(T.diag(y), T.dot(X, beta) + b)
        )
    )
    # Function computing in one go the cost, its gradient
    # with regard to beta and with regard to the bias.
    cost_grad = theano.function(params,[
        cost,
        T.grad(cost, beta),
        T.grad(cost, b)
    ])

    # Function for computing element-wise sigmoid, used for
    # prediction.
    log_predict = theano.function(
        [beta, b, X],
        T.nnet.sigmoid(b + T.dot(X, beta)),
        on_unused_input='warn'
    )

    return (cost_grad, log_predict)

log_cost_grad, log_predict = _compile_func()

class LogisticRegression:
    def __init__(self, C, verbose=False):
        self.C = C
        self.verbose = verbose
    
    def fit(self, X, y, nb_iter=100, learning_rate=0.1, batch_size=5):
        nb_samples, nb_features = X.shape
        assert y.size == nb_samples

        # Convert everything to theano's floatX.
        X_ = X.astype(theano.config.floatX)
        y_ = y.astype(theano.config.floatX)

        # Initial the model to all zeros, as well as the bias.
        beta = np.zeros([nb_features])
        bias = 0

        nb_batches = max(1, nb_samples / batch_size)
        # Thresholds for each mini batch.
        thresh = np.round(
            np.linspace(0, nb_samples, num=nb_batches + 1)
        ).astype(np.int32)

        # Simple mini-batch SGD optimization of the cost function.
        for t in range(1, nb_iter+1):
            # Shuffle the dataset.
            idxs = np.random.permutation(nb_samples)
            
            # Iterate over the mini-batches.
            for i in range(nb_batches):
                batch = X_[idxs[thresh[i]:thresh[i + 1]]]
                labels = y_[idxs[thresh[i]:thresh[i + 1]]]
                cost, grad_beta, grad_b = log_cost_grad(
                    beta, bias, batch, labels, self.C
                )
                beta -= learning_rate * grad_beta
                bias -= learning_rate * grad_b

            cost, grad_beta, grad_b = log_cost_grad(
                beta, bias, X_, y_, self.C
            )

            gradnorm = np.linalg.norm(grad_beta)

            if gradnorm < 10E-5 and np.absolute(grad_b) < 10E-5:
                break

            if self.verbose:
                print "Epoch " + repr(t)
                print "Cost: " + repr(cost)
                print "model gradient norm: " + repr(
                    np.linalg.norm(grad_beta)
                )
                print "bias gradient: " + repr(grad_b)
        
        # Save the final iterate.
        self.beta = beta
        self.bias = bias
        # To keep the interface identical to scikit.
        self.coef_ = beta
        self.intercept_ = bias

    def predict_proba(self, X):
        return log_predict(self.beta, self.bias, X)
