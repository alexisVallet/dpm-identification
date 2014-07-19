""" Linearly calibrate output scores of a binary classifier.
"""
import numpy as np
import theano
import theano.tensor as T
from sklearn.linear_model import LogisticRegression

def _compile_funcs():
    a = T.scalar('a')
    b = T.scalar('b')
    p = T.vector('p')

    return theano.function(
        [a, b, p],
        T.nnet.sigmoid(10 * (p - a) / (b - a) - 5)
    )

log_calibrate = _compile_funcs()

class LogisticCalibrator:
    """ Calibrates score by training a logistic function on the output
        probabilities of the training set. The calibration simply maps
        the probabilities linearly using the info from training data,
        then uses this same linear projection for test data before
        feeding it to a sigmoid for the [0;1] range.
    """
    def __init__(self, classifier, verbose=False):
        self.classifier = classifier
        self.verbose = verbose

    def train(self, positives, negatives):
        # Train the classifier.
        self.classifier.train(positives, negatives)
        # Computes predicted probabilities on the training set.
        probas = self.classifier.predict_proba(positives + negatives)
        self.minprob = probas.min()
        self.maxprob = probas.max()

        if self.verbose:
            print "Training:"
            print "Uncalibrated: "
            print "min: " + repr(self.minprob)
            print "max: " + repr(self.maxprob)
            print "avg: " + repr(probas.mean())
        
        if self.verbose:
            calibrated = log_calibrate(
                self.minprob, 
                self.maxprob, 
                probas
            )
            print "Calibrated:"
            print "min: " + repr(calibrated.min())
            print "max: " + repr(calibrated.max())
            print "avg: " + repr(calibrated.mean())
    
    def predict_proba(self, samples):
        # Compute non-calibrated probabilities from the inner classifier.
        probas = self.classifier.predict_proba(samples)
        if self.verbose:
            print "Testing:"
            print "Uncalibrated: "
            print "min: " + repr(probas.min())
            print "max: " + repr(probas.max())
            print "avg: " + repr(probas.mean())
        # Calibrate them using a sigmoid function.
        calibrated = log_calibrate(
            self.minprob, 
            self.maxprob, 
            probas
        )
        if self.verbose:
            print "Calibrated: "
            print "min: " + repr(calibrated.min())
            print "max: " + repr(calibrated.max())
            print "avg: " + repr(calibrated.mean())
        return calibrated
