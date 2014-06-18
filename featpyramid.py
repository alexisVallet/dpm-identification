""" Feature pyramids for anime character identification.
"""
import cv2
import numpy as np
import skimage.transform as trans
import math
import sys

def computefeaturemap(image, mindimdiv, feature, featdim):
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
    if rotated:
        featuremap = np.transpose(featuremap, (1,0,2))
    return featuremap

class FeatPyramid:
    def __init__(self, image, feature, featdim, mindimdiv=7):
        """ Creates a feature pyramid with 2 layers, one full-resolution, the other
            half resolution, with nbcells for the half resolution one and nbcells*2
            for the full resolution one. Builds square cells with color histogram 
            features.
        
        Arguments
            image     the color image to compute a feature pyramid from.
            feature   arbitrary feature function, taking a color cell from the original
                      image, and returning a featdim-dimensional vector.
            featdim   dimensionality of the features returned by the feature function.
            mindimdiv number of time the smallest dimension of the image should be split
                      in the feature map for the low-resolution layer.
        """
        # Compute full-resolution feature map
        self.features = []
        self.features.append(computefeaturemap(image, mindimdiv * 2, feature, featdim))
        # And half-resolution feature map
        self.features.append(computefeaturemap(cv2.resize(image, None, fx=0.5, fy=0.5,
                                                          interpolation=cv2.INTER_CUBIC)
                                               ,mindimdiv, feature, featdim))

def colorhistogram(image, nbbins=(4,4,4), limits=(0,255,0,255,0,255)):
    """ Compute a color histogram of an image in a numpy array.
    
    Arguments:
        image    color image to compute the color histogram from.
        nbbins   tuple with nbchannels elements specifying bins to use for each channel.
                 Defines the shape of the output histogram, for a total of prod(nbbins)
                 bins.
        limits   tuple with nbchannels elements specifying bounds for color values for
                 each channel. Each element is a pair (minval, maxval) where minval
                 is the lowest possible value (inclusive) and maxval is the highest
                 possible (exclusive).

    Returns:
        An nbbins shaped numpy array containing the color histogram of the input image.
    """
    nbchannels = image.shape[2]

    return cv2.calcHist([image], range(0, nbchannels), None, nbbins, limits)

""" Computes lab histogram of an image.
"""
labhistogram = lambda img, nbbins: colorhistogram(img, nbbins,
                                                  [0,101,-127, 128, -127, 128])

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    nbbins = (4,4,4)
    pyramid = FeatPyramid(img, lambda img: labhistogram(img, nbbins).flatten('C'), 64)
