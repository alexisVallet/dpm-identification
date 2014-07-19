""" Trains a probabilistic binary classifier's parameters through grid 
    search, maximizing the area under the ROC curve on a validation set.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

class BinaryGridSearch:
    def __init__(self, fit_classifier, args):
