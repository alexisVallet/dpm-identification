""" Feature pyramids for anime character identification.
"""
import cv2
import numpy as np
import skimage.transform as trans
import math
import sys
import features as feat
import dpm

class FeatPyramid:
    def __init__(self, image, feature, featdim, mixturesize, mindimdiv=7):
        """ Creates a feature pyramid with 2 layers, one full-resolution, 
            the other half resolution, with nbcells for the half 
            resolution one and nbcells*2 for the full resolution one. 
            Builds (roughly) square cells.
        
        Arguments
            image     the color image to compute a feature pyramid from.
            feature   arbitrary feature function, taking a color cell from
                      the original image, and returning a 
                      featdim-dimensional vector.
            featdim   dimensionality of the features returned by the 
                      feature function.
            mindimdiv number of time the smallest dimension of the image 
                      should be split in the feature map for the 
                      low-resolution layer.
        """
        # Compute full-resolution feature map for parts
        self.features = []
        self.features.append(
            feat.compute_regular_featmap(
                image, mindimdiv * 2, feature, featdim
            )
        )
        # And "half-resolution" feature maps for each root filter
        self.rootfeatures = []
        for dpmsize in mixturesize.dpmsizes:
            self.rootfeatures.append(
                feat.compute_featmap(
                    cv2.resize(image, None, fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_CUBIC),
                    dpmsize.rootshape[0],
                    dpmsize.rootshape[1],
                    feature, 
                    featdim)
            )
