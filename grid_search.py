""" Trains a probabilistic binary classifier's parameters through grid 
    search, maximizing the area under the ROC curve for
"""
import numpy as np
from sklearn.metrics import roc_auc_score

class BinaryGridSearch:
    def __init__(self, fit_classifier, args, k):
        """ Initializes a grid search classifier with a function to
            fit the classifier, and a dictionary of possible argument
            values. Will exhaustively consider all possible combination
            of arguments.

        Arguments:
            fit_classifier
                function taking as input a dictionary of arguments. e.g.,
                if args = { 'C': [1, 0.1, 0.01], 'alpha': [0.1, 0.01] },
                then function must take as input { 'C': x, 'alpha': y } where
                x is in [1, 0.1, 0.01] and y is in [0.1, 0.01], and return
                a classifier to fit (implements a fit(positives, negatives)
                and a predict_proba(samples) method).
            args
                dictionary of classifier parameters to consider, in the
                format described in the doc for fit_classifier.
            k
                number of folds of k-fold cross validation to consider.
        """
        
