""" Features for use in DPMs for character identification.
"""
import cv2
import numpy as np

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

def labhistogram(nbbins):
    """ Computes lab histogram of an image, with each bin normalized to 
        the [0;1] range, as a feature vector. Curried arguments.
    
    Arguments:
        labimg LAB image to compute the color histogram of. Should be a 
               float32 LAB image as defined by OpenCV's cvtColor 
               documentation, i.e. L is between 0 and 100, a and b 
               between -127 and 127 (all bounds inclusive).
        nbbins 3-tuple containing the number of bins per channels.

    Returns:
        A feature vector of size np.prod(nbbins) where each bin is 
        normalized to the [0;1] range - e.g. each bin contains a 
        probability of a random pixel in the patch belonging to the bin 
        (so it all sums to 1).
    """
    return (lambda img: (
        colorhistogram(
            img, 
            nbbins, 
            [0,101,-127, 128, -127, 128]).astype(np.float64)
        / np.prod(img.shape[0:2])).flatten('C'))

def labhistvis(nbbins):
    return lambda vhist: labhistvis_(vhist.reshape(nbbins, order='C'))

def labhistvis_(hist):
    """ Visualize a lab histogram as linear combination of bin median 
        colors weighted by occurences. Returns a lab image.
    """
    lbins, abins, bbins = hist.shape
    lbounds, abounds, bbounds = [(0,100), (-127, 127), (-127,127)]
    # precompute lower and higher bounds for each bin
    lvals, avals, bvals = map(
        lambda ((low,high),bins): np.linspace(low, high, bins+1),
        zip([lbounds, abounds, bbounds], [lbins, abins, bbins])
    )

    featcolor = np.zeros([3], np.float32)
    
    for li in range(0,lbins):
        medl = (lvals[li] + lvals[li+1])/2
        for ai in range(0, abins):
            meda = (avals[ai] + avals[ai+1])/2
            for bi in range(0, bbins):
                medb = (bvals[bi] + bvals[bi+1])/2
                featcolor = (
                    featcolor + 
                    hist[li,ai,bi] * 
                    np.array([medl, meda, medb], np.float32)
                )

    # return a single pixel image (will be resized by the vizualisation
    # procedure for a full feature map)
    return featcolor.reshape([1,1,3])

def visualize_featmap(featuremap, featvis, blocksize=(32,32), 
                       dtype=np.float32):
    """ Returns a visualization for a feature map as a color image,
        given a feature visualization function to apply to each feature.
    """
    brows, bcols = blocksize
    ftrows, ftcols = featuremap.shape[0:2]
    outimage = np.empty([ftrows*brows, ftcols*bcols, 3], dtype=dtype)

    for i in range(0,ftrows):
        for j in range(0,ftcols):
            outimage[i*brows:(i+1)*brows,
                     j*brows:(j+1)*brows] = (
                         cv2.resize(
                             featvis(featuremap[i,j]),
                             blocksize[::-1]
                         )
                     )
    
    return outimage
