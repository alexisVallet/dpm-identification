""" Implements score calibration from multiple LSVMs,
    using Platt's method.
"""
import sklearn.linear_model as sklinear
import numpy as np

import matching

def train_calibrator(model, pospyr, negpyr):
    """ Trains a sigmoid function to calibrate scores for a
        mixture model.
    
    Arguments:
        model         mixture model to calibrate.
        pospyr        positive feature pyramids for the model.
        negpyr        negative feature pyramids for the model.

    Returns:
       A sklearn.linear_model.LogisticRegression object whose
       predict_proba function has been fitted to output probability
       estimates from model scores.
    """
    # Compute the scores using the matching algorithm on
    # each training sample. Store results in a format suitable
    # for scikit-learn.
    nb_pos = len(pospyr)
    nb_neg = len(negpyr)
    nb_samples = nb_pos + nb_neg
    scores = np.empty([nb_samples], np.float64)
    labels = np.empty([nb_samples], np.float64)
    modelvec = model.tovector()
    
    i = 0
    for pos in pospyr:
        (score_, c, latvec) = matching.mixture_matching(pos, model)
        scores[i] = np.vdot(modelvec, latvec)
        labels[i] = 1
        i = i + 1
    for neg in negpyr:
        (score_, c, latvec) = matching.mixture_matching(neg, model)
        scores[i] = np.vdot(modelvec, latvec)
        labels[i] = -1
        i = i + 1

    # fit the logistic function
    logregr = sklinear.LogisticRegression(C=10)
    logregr.fit(scores, labels)
    
    return logregr
