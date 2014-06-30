""" Simple test classifier for latent logistic regression. Matches a single part
    to the feature maps, with part position as latent values. The part is free moving,
    i.e. there is no deformation cost or anchor position.
"""
import numpy as np
import cv2

from matching import match_filter
from latent_lr import BinaryLLR
from features import compute_regular_featmap
from warpclassifier import WarpClassifier

class SinglePartClassifier:
    def __init__(self, C, feature, mindimdiv, verbose=False):
        self.C = C
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.verbose = verbose

    def best_match(self, partmodel, featmap):
        """ Returns the flattened subwindow of the feature map which maps the part 
            filter best.
        """
        print partmodel.shape
        print repr(self.partsize * self.partsize * self.feature.dimension)
        part = partmodel.reshape(self.partsize, self.partsize, self.feature.dimension)
        (response, padded) = match_filter(featmap, part)
        maxi, maxj = np.unravel_index(
            np.argmax(response),
            response.shape
        )
        
        return padded[maxi:maxi+self.partsize,
                      maxj:maxj+self.partsize].flatten('C')

    def train(self, positives, negatives):
        """ Fits the classifier a set of positive images and a set of negative
            images.
        
        Arguments:
            positives    array of positive images, i.e. the classifier should return
                         a probability close to 1 for.
            negatives    array of negative images, i.e. the classifier should return
                         a probability close to 0 for.
        """
        # Initialize the model by training a warping classifier on all images,
        # and taking the highest energy subwindow of the corresponding model with
        # a small square patch size (half mindimdiv).
        warp = WarpClassifier(self.feature, self.mindimdiv, self.C)
        warp.train(positives, negatives)
        warpmap = warp.model_featmap
        wrows, wcols, featdim = warpmap.shape
        self.partsize = self.mindimdiv // 2
        maxsubwin = None
        maxenergy = 0

        for i in range(wrows - self.partsize):
            for j in range(wcols - self.partsize):
                subwin = warpmap[i:i+self.partsize,j:j+self.partsize]
                energy = np.vdot(subwin, subwin)
                if maxsubwin == None or maxenergy < energy:
                    maxenergy = energy
                    maxsubwin = subwin
        initpart = maxsubwin.flatten('C')

        # Compute feature maps for all samples
        def tofeatmap(image):
            return compute_regular_featmap(image, self.feature, self.mindimdiv)

        posmaps = map(tofeatmap, positives)
        negmaps = map(tofeatmap, negatives)
        
        # Train a latent logistic regression on the feature maps with the
        # best match latent function.
        self.llr = BinaryLLR(self.best_match, self.C, self.verbose)
        self.llr.fit(posmaps, negmaps, initpart, 4)
