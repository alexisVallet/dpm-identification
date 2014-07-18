""" Linearly calibrate output scores of a binary classifier.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticCalibrator:
    """ Calibrates score by training a logistic function on the output
        probabilities of the training set.
    """
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, positives, negatives):
        # Train the classifier.
        self.classifier.train(positives, negatives)
        # Computes predicted probabilities on the training set.
        probas = self.classifier.predict_proba(positives + negatives)
        # Train a logistic function to calibrate scores.
        labels = np.empty([probas.size])
        labels[0:len(positives)] = 1
        labels[len(positives):] = 0
        self.lr = LogisticRegression(C=10)
        print probas.shape
        self.lr.fit(np.transpose(probas), labels)
    
    def predict_proba(self, samples):
        # Compute non-calibrated probabilities from the inner classifier.
        probas = self.classifier.predict_proba(samples)
        # Calibrate them using the previously trained logistic function.
        return self.lr.predict_proba(probas)[:,1]
        
