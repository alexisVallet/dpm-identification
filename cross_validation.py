""" Cross validation for multi-class classifiers.
"""
import numpy as np

class CrossValidator:
    def __init__(self, k, classifier):
        """ Initializes a k-fold cross validator for a multi-class
            classifier.
        
        Arguments:
            k
                the number of folds of cross validation.
            classifier
                the classifier to perform cross validation on.
        """
        self.k = k
        self.classifier = classifier

    def train(self, samples, labels):
        
