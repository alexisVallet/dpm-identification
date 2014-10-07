""" General utilities for features.
"""
import cv2
import numpy as np
import theano
from sklearn.decomposition import PCA
from skimage import exposure

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

class Combine(Feature):
    """ Combines features into one by concatenating feature maps.
    """
    def __init__(self, *features):
        for f in features:
            assert isinstance(f, Feature)
        self.features = features
        self.dimension = sum(map(lambda f: f.dimension, features))

    def compute_featmap(self, image, n, m):
        fmaps = map(lambda f: f.compute_featmap(image, n, m),
                    self.features)
        return np.concatenate(fmaps, axis=2)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

def uncompress(image):
    if len(image.shape) < 2:
        return cv2.imdecode(image, 1)
    else:
        return image
    
class BGRHist(Feature):
    """ Computes flattened and scaled BGR histograms as features.
    """
    def __init__(self, nbbins, block_halfsize):
        """ Initializes the feature with a number of bins per channels
            as a triplet.
        """
        self.nbbins = nbbins
        self.dimension = np.prod(self.nbbins) * (block_halfsize * 2 + 1)**2
        self.block_halfsize = block_halfsize

    def compute_featmap(self, image, n, m):
        """ Compute a feature map of flattened color histograms for each
            block in the image.
        """
        cells= np.empty(
            [n + self.block_halfsize*2, m + self.block_halfsize*2, 
             np.prod(self.nbbins)],
            dtype=theano.config.floatX
        )
        # Uncompress if necessary.
        _image = uncompress(image)
        # Limits of color values. Takes the OpenCV convention:
        # 0-255 for uint8, [0;1] for float32.
        assert _image.dtype in [np.uint8, np.float32]
        limits = (0,256) * 3 if _image.dtype == np.uint8 else (0.,1.) * 3

        for _cell in block_generator(_image, 
                                     n + self.block_halfsize*2, 
                                     m + self.block_halfsize*2):
            i, j, cell = _cell
            hist = cv2.calcHist([cell], range(3), None, self.nbbins,
                                limits).astype(theano.config.floatX)
            feature = None
            feature = hist.flatten('C') / hist.max()
            cells[i,j] = feature

        return block_normalization(cells, self.block_halfsize)

    def visualize_featmap(self, fmap):
        return visualize_featmap(fmap, bgrhistvis)

    def __repr__(self):
        return "BGRHist(%r,%r)" % (self.nbbins, self.block_halfsize)

class HoG(Feature):
    """ Computes HoG features, as described by Dalal and Triggs, 2005. Much
        of the code was inspired by scikit image's HoG implementation.
    """
    def __init__(self, nb_orient, block_halfsize):
        """ Initialize the HoG features for a number of orientation histogram
            bins, and a given normalization block half size.
        """
        self.nb_orient = nb_orient
        self.block_halfsize = block_halfsize
        self.dimension = nb_orient * (block_halfsize * 2 + 1)**2

    def compute_featmap(self, comp_image, n, m):
        """ Computes a feature map of flattened HoG features. The returned
            feature map is a map of blocks, which are themselves multiple
            concatenated cells normalized together.
        """
        # Uncompress image if necessary.
        image = uncompress(comp_image)
        assert image.dtype in [np.uint8, np.float32]
        # Convert image to grayscale, floating point.
        _image = None
        if image.dtype != np.float32:
            _image = image.astype(np.float32) / 255
        else:
            _image = image
        gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        # Compute horizontal and vertical gradients.
        gx = cv2.filter2D(gray, -1, np.array([-1, 0, 1]).reshape((1,3)))
        gy = cv2.filter2D(gray, -1, np.array([-1, 0, 1]))
        # Compute unsigned gradient orientation map.
        orient = np.arctan2(gx, gy) % np.pi
        # Then compute histograms for each cell of this gradient
        # orientation map. We compute n+block size by m+block size
        # cells so the block normalization process produces n by m
        # features in the end.
        cells = np.empty(
            [n+2*self.block_halfsize, 
             m+2*self.block_halfsize, 
             self.nb_orient],
            dtype=theano.config.floatX
        )
        # Masking orientations whose gradient magnitude is below some
        # threshold.
        mag_eps = 10E-3
        mag_mask = (np.sqrt(gx**2 + gy**2) > mag_eps).astype(np.uint8)
        for _cell in block_coord_generator(orient.shape, 
                                           n+2*self.block_halfsize, 
                                           m+2*self.block_halfsize):
            i, j, (starti, endi), (startj, endj) = _cell
            cell_orient = orient[starti:endi,startj:endj]
            cell_mask = mag_mask[starti:endi,startj:endj]
            hist = cv2.calcHist(
                [cell_orient],
                [0],
                cell_mask,
                [self.nb_orient],
                (0, np.pi)
            ).reshape((self.nb_orient,))
            cells[i,j] = hist
        
        return block_normalization(cells, self.block_halfsize)

    def visualize_featmap(self, fmap):
        block_size = self.block_halfsize*2+1
        assert fmap.shape[2] == self.nb_orient * block_size**2
        cellsize = 16
        rows, cols = fmap.shape[0:2]
        radius = cellsize // 2 - 1
        hog_image = np.zeros(
            [rows*cellsize*block_size, 
             cols*cellsize*block_size], 
            np.float32
        )
        orientations = np.linspace(0, np.pi, num=self.nb_orient+1)

        for i in range(rows):
            for j in range(cols):
                block = fmap[i,j].reshape(
                    [block_size, block_size, self.nb_orient]
                )
                for bi in range(block_size):
                    for bj in range(block_size):
                        for o in range(self.nb_orient):
                            med_orient = (
                                orientations[o] + orientations[o+1]
                            ) / 2
                            cx, cy = (bj*cellsize + j*block_size*cellsize 
                                      + cellsize // 2,
                                      bi*cellsize + i*block_size*cellsize 
                                      + cellsize // 2)
                            dx = int(radius * np.cos(med_orient))
                            dy = int(radius * np.sin(med_orient))
                            cv2.line(hog_image,
                                     (cx - dx, cy - dy), (cx + dx, cy + dy),
                                     float(block[bi,bj,o]))
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0,0.2))
        return hog_image

    def __repr__(self):
        return "HoG(%r, %r)" % (self.nb_orient, self.block_halfsize)

def compute_pyramids(samples, max_dims, feature, min_var=None):
    """ Compute feature pyramids for a bunch of samples, optionally applying
        dimensionality reduction to the features. Returns them as lists of
        feature maps for each sample.
    """
    nb_samples = len(samples)
    nb_scales = len(max_dims)
    # Compute all the feature maps at each scale.
    fmaps_per_scale = []
    
    for max_dim in max_dims:
        fmaps, rows, cols = warped_fmaps_simple(samples, max_dim, feature)
        fmaps_per_scale.append(fmaps)

    return fmaps_per_scale
     
def block_generator(image, n, m):
    """ Returns a generator which iterates over non-overlapping blocks
        in a n by m grid in the input image. Blocks will be as close to
        equal sized as possible.
    """
    for block_coords in block_coord_generator(image.shape, n, m):
        (i, j, (starti,endi), (startj, endj)) = block_coords
        yield (i, j, image[starti:endi, startj:endj])

def block_coord_generator(shape, n, m):
    rows, cols = shape[0:2]
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
            yield (i, j, (starti, endi), (startj, endj))

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

    for i in range(wrows - winsize + 1):
        for j in range(wcols - winsize + 1):
            subwin = featmap[i:i+winsize,j:j+winsize]
            energy = np.vdot(subwin, subwin)
            if maxsubwin == None or maxenergy < energy:
                maxanchor = (i,j)
                maxenergy = energy
                maxsubwin = subwin

    return (np.array(maxsubwin, copy=True), maxanchor)

def warped_fmaps_dimred(samples, mindimdiv, feature, min_var=0.9):
    """ Compute warped feature maps for a set of samples, applying
        PCA to the features as a preprocessing step to the features.
        Returns the corresponding sklearn PCA object, so further
        data can easily be projected to the new subspace.
    """
    nb_samples = len(samples)
    # Compute all the features.
    fmaps, rows, cols = warped_fmaps_simple(samples, mindimdiv, feature)
    # Slap them into a data matrix.
    X = np.empty(
        [nb_samples * rows * cols, feature.dimension],
        dtype=theano.config.floatX
    )
    for i in range(nb_samples):
        X[i*rows*cols:(i+1)*rows*cols] = np.reshape(
            fmaps[i], [rows * cols, feature.dimension]
        )
    # Run PCA on it.
    pca = PCA(min_var)
    X_ = pca.fit_transform(X)
    new_featdim = X_.shape[1]
    # Slap them into feature maps.
    fmaps_dimred = []

    for i in range(nb_samples):
        fmaps_dimred.append(
            np.reshape(
                X_[i*rows*cols:(i+1)*rows*cols],
                [rows, cols, new_featdim]
            )
        )
    return (fmaps_dimred, rows, cols, pca)

def compute_mean_ar(samples):
    def compute_ar(image):
        # Uncompress image if necessary.
        _image = uncompress(image)
        return (float(_image.shape[1]) / _image.shape[0])
    return np.mean(map(compute_ar, samples))

def warped_fmaps_simple(samples, mindimdiv, feature):
    # Find out the average aspect ratio across
    # positive samples. Use that value to define
    # the feature map dimensions.
    meanar = compute_mean_ar(samples)
    # Basic algebra to get the corresponding number of rows/cols
    # from the desired minimum dimension divisions.
    nbrowfeat = None
    nbcolfeat = None

    if meanar > 1:
        nbrowfeat = mindimdiv
        nbcolfeat = int(mindimdiv * meanar)
    else:
        nbrowfeat = int(mindimdiv / meanar)
        nbcolfeat = mindimdiv
    
    tofeatmap = lambda s: feature.compute_featmap(s, nbrowfeat, nbcolfeat)
    return (map(tofeatmap, samples), nbrowfeat, nbcolfeat)

def warped_fmaps(positives, negatives, mindimdiv, feature):
    """ Computes feature maps warped to the mean positive aspect ratio.

    Arguments:
        positives
            list of positive image samples.
        negatives
            list of negative image samples.
        mindimdiv
            number of division for the minimum dimension of the feature
            maps.
    Returns:
       (posmaps, negmaps, nbrowfeat, nbcolfeat) where posmaps and negmaps
       are feature maps for positive and negative samples respectively,
       and [nbrowfeat, nbcolfeat] are the first 2 dimensions of the feature
       maps (the third is the feature dimension).
    """
    # Find out the average aspect ratio across
    # positive samples. Use that value to define
    # the feature map dimensions.
    meanar = np.mean(map(lambda pos: float(pos.shape[1]) / pos.shape[0],
                         positives))
    # Basic algebra to get the corresponding number of rows/cols
    # from the desired minimum dimension divisions.
    nbrowfeat = None
    nbcolfeat = None

    if meanar > 1:
        nbrowfeat = mindimdiv
        nbcolfeat = mindimdiv * meanar
    else:
        nbrowfeat = int(mindimdiv / meanar)
        nbcolfeat = mindimdiv
    
    tofeatmap = lambda pos: feature.compute_featmap(pos, nbrowfeat, nbcolfeat)
    posmaps = map(tofeatmap, positives)
    negmaps = map(tofeatmap, negatives)
    
    return (posmaps, negmaps, nbrowfeat, nbcolfeat)

def block_normalization(cells, block_halfsize):
    # Apply a block normalization scheme. To fit a data structure
    # compatible with our color histograms, each block is flattened
    # into a single cell of the feature map, so we get a n by m map which
    # is still meaningful.
    rows, cols, fdim = cells.shape
    n = rows - 2 * block_halfsize
    m = cols - 2 * block_halfsize
    blocks = np.empty(
        [n, m, fdim * (block_halfsize*2 + 1)**2],
        dtype=theano.config.floatX
    )
    eps = 10E-5
    
    for i in range(n):
        for j in range(m):
            block = cells[
                i:i + 2*block_halfsize+1,
                j:j + 2*block_halfsize+1
            ].flatten('C')
            block = block / np.sqrt(np.dot(block,block) + eps)
            blocks[i,j] = block
    
    return blocks
