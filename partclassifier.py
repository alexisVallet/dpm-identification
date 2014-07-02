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
    def __init__(self, C, feature, mindimdiv, verbose=False, debug=False,
                 algorithm=None):
        self.C = C
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.verbose = verbose
        self.debug = debug
        self.algorithm = algorithm

    def best_match(self, partmodel, featmap):
        """ Returns the flattened subwindow of the feature map which maps the part 
            filter best.
        """
        part = partmodel.reshape(self.partsize, self.partsize, self.feature.dimension)
        (response, padded) = match_filter(featmap, part, return_padded=True)
        maxi, maxj = np.unravel_index(
            np.argmax(response),
            response.shape
        )
        
        return padded[maxi:maxi+self.partsize,
                      maxj:maxj+self.partsize].flatten('C')

    def tofeatmap(self, image):
        return compute_regular_featmap(image, self.feature, self.mindimdiv)

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
        if self.debug:
            # Displays the learned warped feature map
            image = self.feature.vis_featmap(warpmap)
            cv2.namedWindow("Initial warped map", cv2.WINDOW_NORMAL)
            cv2.imshow("Initial warped map", image)
            cv2.waitKey(0)
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
        posmaps = map(self.tofeatmap, positives)
        negmaps = map(self.tofeatmap, negatives)
        
        # Train a latent logistic regression on the feature maps with the
        # best match latent function.
        self.llr = BinaryLLR(self.best_match, self.C, self.verbose, self.algorithm)
        self.llr.fit(posmaps, negmaps, initpart, 4)
        # For vizualisation, set the featmap to the resahped model vector.
        self.model_featmap = self.llr.model[1:].reshape(
            self.partsize,
            self.partsize,
            self.feature.dimension
        )

    def predict_proba(self, images):
        """ Predicts probabilities that images are positive samples.

        Arguments:
            images    image to predict probabilities for.
        """
        assert self.llr != None
        # Compute feature maps
        fmaps = map(self.tofeatmap, images)
        
        # Use the internal latent logistic regression to predict probabilities.
        return self.llr.predict_proba(fmaps)
