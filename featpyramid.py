""" Feature pyramids for anime character identification.
"""
import cv2
import numpy as np
import skimage.transform as trans
import math
import sys

def compute_featmap(image, n, m, feature, featdim):
    height, width = image.shape[0:2]
    featuremap = np.empty([m, n, featdim])
    # We cut up the image into an m by n grid. To avoir rounding errors, we
    # first compute the points of the grid, then iterate over them.
    rowindexes = np.round(np.linspace(0, height, num=m+1)).astype(np.int32)
    colindexes = np.round(np.linspace(0, width, num=n+1)).astype(np.int32)

    for i in range(0,m):
        starti = rowindexes[i]
        endi = rowindexes[i+1]
        for j in range(0,n):
            startj = colindexes[j]
            endj = colindexes[j+1]
            block = image[starti:endi,startj:endj]
            # Compute its feature
            featuremap[i,j,:] = feature(block)
    return featuremap

def compute_regular_featmap(image, mindimdiv, feature, featdim):
    # Compute the features for each layer. A feature is represented as a 3d
    # numpy array with dimensions w*h*featdim where w and h are the width and
    # height of the input downsampled image.        
    # First compute the feature map for the full-resolution layer
    height, width = image.shape[0:2]
    
    # Split the image across the smallest dimension. We assume the width is the
    # the smallest, if that's not the case we transpose it.
    rotated = min(height,width) == height
    if rotated:
        image = np.transpose(image, (1,0,2))
        height, width = width, height

    # Compute the number of division for the height
    n = mindimdiv
    m = int(round(height * n / width))

    featuremap = compute_featmap(image, n, m, feature, featdim)
    
    if rotated:
        featuremap = np.transpose(featuremap, (1,0,2))

    return featuremap

class FeatPyramid:
    def __init__(self, image, feature, featdim, mindimdiv=7):
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
        # Compute full-resolution feature map
        self.features = []
        self.features.append(
            compute_regular_featmap(image, mindimdiv * 2, feature, featdim)
        )
        # And half-resolution feature map
        self.features.append(
            compute_regular_featmap(
                cv2.resize(image, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_CUBIC)
                ,mindimdiv, feature, featdim)
        )
