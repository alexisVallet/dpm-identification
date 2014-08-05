""" General utilities for features.
"""
import cv2
import numpy as np
import theano

class Feature:
    """ Abstract feature class specifying the interface for all features
        to work well with the DPM / Latent LR framework.
    """
    def compute_featmap(self, image, n, m):
        """ Implementing classes should compute a n by m feature map
            on the source image. Features should be scaled to a
            "small enough" range (ie. [-1;1] or [0;1]) so the learning
            algorithms don't run into scaling issues.
        """
        raise NotImplemented()

class BGRHist(Feature):
    """ Computes flattened and scaled BGR histograms as features.
    """
    def __init__(self, nbbins):
        """ Initializes the feature with a number of bins per channels
            as a triplet.
        """
        self.nbbins = nbbins

    def compute_featmap(self, image, n, m):
        """ Compute a feature map of flattened color histograms for each
            block in the image.
        """
        fmap = np.empty(
            [n, m, np.prod(self.nbbins)],
            dtype=theano.config.floatX
        )
        # Limits of color values. Takes the OpenCV convention:
        # 0-255 for uint8, [0;1] for float32.
        assert image.dtype in [np.uint8, np.float32]
        limits = [(0,255)] * 3 if image.dtype == np.uint8 else [(0,1)] * 3

        for _block in block_generator(image, n, m):
            i, j, block = _block
            fmap[i,j] = colorhistogram(
                self.nbbins,
                limits
            ).flatten('C')

class HoG(Feature):
    """ Computes HoG features, as described by Dalal and Triggs, 2005. Much
        of the code was inspired by scikit image's HoG implementation.
    """
    def __init__(self, nb_orient):
        self.nb_orient = nb_orient

    def compute_featmap(self, image, n, m):
        """ Computes a feature map of flattened HoG features.
        """
        assert image.dtype in [np.uint8, np.float32]
        # Convert image to grayscale, floating point.
        _image = None
        if image.dtype != np.float32:
            _image = image.astype(np.float32) / 255
        else:
            _image = image
        gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        cv2.waitKey(0)
        # Compute horizontal and vertical gradients.
        gx = cv2.filter2D(gray, -1, np.array([-1, 0, 1]).reshape((1,3)))
        gy = cv2.filter2D(gray, -1, np.array([-1, 0, 1]))
        cv2.imshow('gx', gx)
        cv2.imshow('gy', gy)
        cv2.waitKey(0)
        # Compute unsigned gradient orientation map.
        orient = np.arctan2(gx, gy) % np.pi
        cv2.imshow('orient', orient / np.pi)
        cv2.waitKey(0)
        # Then compute histograms for each block of this gradient
        # orientation map.
        fmap = np.empty([n, m, self.nb_orient])
        for _block in block_generator(orient, n, m):
            i, j, block = _block
            hist = cv2.calcHist(
                [block],
                [0],
                None,
                [self.nb_orient],
                (0, np.pi)
            ).reshape((self.nb_orient,))
            # Right now we simply normalize linearly to [0;1] range.
            # The more complex normalizations of true HoG (by block)
            # could be of interest later on though.
            fmap[i,j] = hist.astype(np.float32) / hist.max()
        
        return fmap

    def visualize_featmap(self, fmap):
        assert fmap.shape[2] == self.nb_orient
        cellsize = 32
        rows, cols = fmap.shape[0:2]
        radius = cellsize // 2 - 1
        hog_image = np.zeros([rows*cellsize, cols*cellsize], np.float32)
        orientations = np.linspace(0, np.pi, num=self.nb_orient+1)

        for i in range(rows):
            for j in range(cols):
                for o in range(self.nb_orient):
                    med_orient = (orientations[o] + orientations[o+1]) / 2
                    cx, cy = (j*cellsize + cellsize // 2,
                              i*cellsize + cellsize // 2)
                    dx = int(radius * np.cos(med_orient))
                    dy = int(radius * np.sin(med_orient))
                    cv2.line(hog_image[i*cellsize:(i+1)*cellsize,
                                       j*cellsize:(j+1)*cellsize],
                             (cx - dx, cy - dy), (cx + dx, cy + dy),
                             fmap[i,j,o])
        return hog_image

def block_generator(image, n, m):
    """ Returns a generator which iterates over non-overlapping blocks
        in a n by m grid in the input image. Blocks will be as close to
        equal sized as possible.
    """
    rows, cols = image.shape[0:2]
    rowidxs = np.round(
        np.linspace(0, rows, num=n+1)
    ).astype(np.int32)
    colidxs = np.round(
        np.linspace(0, cols, num=m+1)
    ).astype(np.int32)

    for i in range(n):
        starti = rowidxs[i]
        endi = rowidxs[i+1]
        for j in range(m):
            startj = colidxs[j]
            endj = colidxs[j+1]
            yield (i, j, image[starti:endi,startj:endj])

def colorhistogram(image, nbbins=(4,4,4), limits=(0,255,0,255,0,255)):
    """ Compute a color histogram of an image in a numpy array.
    
    Arguments:
        image    color image to compute the color histogram from.
        nbbins   tuple with nbchannels elements specifying bins to use 
                 for each channel. Defines the shape of the output 
                 histogram, for a total of prod(nbbins) bins.
        limits   tuple with nbchannels elements specifying bounds for 
                 color values for each channel. Each element is a pair 
                 (minval, maxval) where minval is the lowest possible 
                 value (inclusive) and maxval is the highest possible 
                 (exclusive).

    Returns:
        An nbbins shaped numpy array containing the color histogram of 
        the input image.
    """
    nbchannels = image.shape[2]
    
    return cv2.calcHist([image], range(0, nbchannels), None, nbbins, 
                        limits)
