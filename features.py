""" Features for use in DPMs for character identification.
"""
import cv2
import numpy as np

class Feature:
    """ Picklable feature class, encapsulating all the necessary
        information to build feature maps from an image.
    """
    _featnames = [
        'labhist', # L*a*b* color histogram
        'bgrhist'  # BGR color histogram
    ]

    def __init__(self, featname, dimension, params):
        self.featname = featname
        self.dimension = dimension
        self.params = params

    def __call__(self, image):
        _featfuncs = {
            'labhist': labhistogram,
            'bgrhist': bgrhistogram
        }
        return _featfuncs[self.featname](self.params)(image)

    def visualize(self, featvector):
        _visfuncs = {
            'labhist': labhistvis,
            'bgrhist': bgrhistvis,
        }
        return _visfuncs[self.featname](self.params)(featvector)

    def vis_featmap(self, featmap):
        return visualize_featmap(
            featmap,
            self.visualize,
            blocksize=(1,1)
        )

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

def compute_featmap(image, n, m, feature):
    height, width = image.shape[0:2]
    featuremap = np.empty([n, m, feature.dimension])
    # We cut up the image into an m by n grid. To avoir rounding errors, we
    # first compute the points of the grid, then iterate over them.
    rowindexes = np.round(np.linspace(0, height, num=n+1)).astype(np.int32)
    colindexes = np.round(np.linspace(0, width, num=m+1)).astype(np.int32)

    for i in range(0,n):
        starti = rowindexes[i]
        endi = rowindexes[i+1]
        for j in range(0,m):
            startj = colindexes[j]
            endj = colindexes[j+1]
            block = image[starti:endi,startj:endj]
            # Compute its feature
            featuremap[i,j,:] = feature(block)
    return featuremap

def regular_grid(image, mindimdiv):
    """ Computes the number of row divisions and column divisions
        necessary to get blocks that are roughly square, given
        that the smallest dimension of the image should be divided
        by mindimdiv.
    """
    height, width = image.shape[0:2]
    
    # Split the image across the smallest dimension. We assume the width is the
    # the smallest, if that's not the case we transpose it.
    rotated = min(height,width) == height
    if rotated:
        height, width = width, height

    # Compute the number of division for the height
    n = mindimdiv
    m = int(round(height * n / width))

    return [n,m] if rotated else [m,n]

def compute_regular_featmap(image, feature, mindimdiv):
    # Compute the features for each layer. A feature is represented as a 3d
    # numpy array with dimensions w*h*featdim where w and h are the width and
    # height of the input downsampled image.        
    # First compute the feature map for the full-resolution layer
    [n,m] = regular_grid(image, mindimdiv)

    featuremap = compute_featmap(image, n, m, feature)

    return featuremap

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

def bgrhistogram(nbbins):
    def bgrhistogram_(img):
        hist = colorhistogram(
            img,
            nbbins,
            [0,255,0,255,0,255])
        return (hist.astype(np.float64) / np.prod(img.shape[0:2])).flatten('C')
    return bgrhistogram_

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

def np_labhistogram(nbbins):
    return (lambda img: (
        np.histogramdd(
            img.reshape([img.shape[0] * img.shape[1], 3], order='C'),
            nbbins,
            [(0,100), (-127, 127), (-127, 127)])[0].astype(np.float64)
        / np.prod(img.shape[0:2])).flatten('C'))

def histvis(bounds, hist_):
    # As the histogram may be the result of (L)SVM training, individual
    # values may be negative. A negative feature is a feature that
    # contributes more to the other characters than the one we're
    # interested in, so we drop it. We then take the square of each
    # feature.
    hist = hist_.copy()
    hist[hist < 0] = 0
    hist = np.multiply(hist, hist)
    lbins, abins, bbins = hist.shape
    lbounds, abounds, bbounds = bounds
    # precompute lower and higher bounds for each bin
    lvals, avals, bvals = map(
        lambda ((low,high),bins): np.linspace(low, high, bins+1),
        zip([lbounds, abounds, bbounds], [lbins, abins, bbins])
    )

    featcolor = np.zeros([3], np.float32)
    sumbins = 0.
    
    for li in range(0,lbins):
        medl = (lvals[li] + lvals[li+1])/2
        for ai in range(0, abins):
            meda = (avals[ai] + avals[ai+1])/2
            for bi in range(0, bbins):
                # ignore negative coefficients
                medb = (bvals[bi] + bvals[bi+1])/2
                bincolor = np.array([medl, meda, medb], np.float32)
                featcolor = featcolor + (hist[li,ai,bi] * bincolor)
                sumbins += hist[li,ai,bi]

    if sumbins != 0:
        featcolor /= sumbins

    # return a single pixel image (will be resized by the vizualisation
    # procedure for a full feature map)
    return featcolor.reshape([1,1,3])

def labhistvis(nbbins):
    return lambda vhist: histvis(
        [(0,100), (-127, 127), (-127, 127)],
        vhist.reshape(nbbins, order='C'))

def bgrhistvis(nbbins):
    return lambda vhist: histvis(
        [(0,1), (0,1), (0,1)],
        vhist.reshape(nbbins, order='C'))

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
                             blocksize[::-1],
                             interpolation=cv2.INTER_NEAREST
                         )
                     )
    
    return outimage

def max_energy_subwindow(featmap, winsize):
    """ Compute and return the highest energy subwindow of a given
        square size in a feature map.

    Arguments:
        featmap    feature map to compute the highest energy subwindow
                   of.
        winsize    size of the window, i.e. the window will have size
                   winsize columns by winsize rows.
    Returns:
        (maxsubwin, maxanchor) where maxsubwin is the subwindow of
        maximum energy in the feature map, and maxanchor is the position
        of the top-left of the window in the original feature map.
    """
    wrows, wcols, featdim = featmap.shape
    maxanchor = None
    maxsubwin = None
    maxenergy = 0
    
    for i in range(wrows - winsize):
        for j in range(wcols - winsize):
            subwin = featmap[i:i+winsize,j:j+winsize]
            energy = np.vdot(subwin, subwin)
            if maxsubwin == None or maxenergy < energy:
                maxanchor = (i,j)
                maxenergy = energy
                maxsubwin = subwin
    
    return (np.array(maxsubwin, copy=True), maxanchor)
