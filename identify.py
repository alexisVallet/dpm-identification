""" Identification algorithm for the binary classification problem.
Essentially implements the DPM matching algorithm by Felzenszwalb and
Girshick.
"""
import cv2
import numpy as np
import scipy.ndimage.filters as filters
import gdt
import dpm
import featpyramid as fpyr

def identify(pyramid, model):
    """ Returns the score of a mixture of dpms on a feature pyramid.

    Arguments:
        pyramid    pyramid to match the model on.
        model      model to match on the feature pyramid.

    Returns:
        The score of the best match for the model on the pyramid.
    """
    # Match each component of the mixture one by one
    for dpm in model.dpms:
        # Compute the response of all the filters of the dpm one by one
        # by cross correlation. The root filter is correlated to the low
        # resolution layer, and the parts to the high resolution one.
        rootresponse = filters.correlate(pyramid.features[1], dpm.root)
        partresponses = map(lambda part: 
                            filters.correlate(pyramid.features[0],part),
                            dpm.parts)
        # Apply distance transform to part responses
        gdtpartresp = map(lambda partresp:
                          gdt.distancetransform(dpm.deform, partresp),
                          partresponses)
