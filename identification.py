""" Multi-class identification.
"""
import numpy as np
import cv2
import sklearn.linear_model as sklinear

import matching
import featpyramid as pyr
import model

def binary_identification(binmodel, images, calibrator=None):
    """ Returns the score of a binary mixture model on a
        list of images.
    """
    modelsize = binmodel.mixture.size()
    modelvector = binmodel.mixture.tovector()

    rawscores = np.empty([len(images)])
    i = 0
    for image in images:
        # compute the pyramid for the image
        pyramid = pyr.FeatPyramid(image, binmodel.feature, binmodel.featdim, 
                                  modelsize, binmodel.mindimdiv)
        # match the model against the pyramid
        score, c, latvec = matching.mixture_matching(pyramid, 
                                                     binmodel.mixture)
        print "dot score: " + repr(np.vdot(modelvector, latvec))
        print "match score: " + repr(score)
        rawscores[i] = score

    # calibrate the scores to the [0;1] range
    return binmodel.calibrator.predict_proba(rawscores)
