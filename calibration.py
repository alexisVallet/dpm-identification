""" Linearly calibrate output scores of a binary classifier.
"""
import numpy as np
import theano
import theano.tensor as T
from sklearn.linear_model import LogisticRegression

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
        minprob = probas.min()
        maxprob = probas.max()

        if self.verbose:
            print "Training:"
            print "Uncalibrated: "
            print "min: " + repr(minprob)
            print "max: " + repr(maxprob)
            print "avg: " + repr(probas.mean())

        p = T.vector('p')
        # Compile the theano function for calibration.
        self.calibrate = theano.function(
            [p],
            T.nnet.sigmoid(10 * (p - minprob) / (maxprob - minprob) - 5)
        )
        if self.verbose:
            calibrated = self.calibrate(probas)
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
        calibrated = self.calibrate(probas)
        if self.verbose:
            print "Calibrated: "
            print "min: " + repr(calibrated.min())
            print "max: " + repr(calibrated.max())
            print "avg: " + repr(calibrated.mean())
        return calibrated
