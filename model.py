""" Model class encompassing a model and all parameters in a picklable
    form.
"""
import numpy as np
import cv2
import sklearn.linear_model as sklinear

import dpm
import features as feat

class Feature:
    """ Enum of features. Only really necessary because
        serializing python functions is a mess.
    """
    bgrhistogram = 0
    labhistogram = 1

def featurefunc(feature):
    """ Returns the function corresponding to a 
        given feature enum.
    """
    feattofunc = {
        Feature.bgrhistogram: feat.bgrhistogram,
        Feature.labhistogram: feat.labhistogram
    }

    return feattofunc[feature]

class BinaryModel:
    """ Model for a binary classifier.
    """
    def __init__(self, mixture, calibrator, feature, featparams,
                 featdim, mindimdiv):
        self.mixture = mixture
        self.feature_ = feature
        self.featparams = featparams
        self.featdim = featdim
        self.mindimdiv = mindimdiv
        self.calibrator = calibrator

    def feature(self):
        return featurefunc(self.feature)(self.featparams)

class MultiModel:
    """ Model for a multi-class classifier.
    """
    def __init__(self, binmodels):
        # check that everything is in order
        assert len(binmodels) > 0
        self.feature = None
        self.featparams = None
        self.featdim = None
        for binmodel in binmodels:
            if self.feature == None:
                self.feature = binmodel.feature
                self.featparams = binmodel.featparams
                self.featdim = binmodel.featdim
            else:
                assert binmodel.feature == self.feature
                assert binmodel.featparams == self.featparams
                assert binmodel.featdim == self.featdim
        self.binmodels = binmodels

    def feature(self):
        return featurefunc(self.feature)(self.featparams)
