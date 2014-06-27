""" Simple linear classifier for warped "root" feature maps over images.
"""
from sklearn.linear_model import LogisticRegression
import numpy as np

import features as feat

class WarpClassifier:
    def __init__(self, feature, mindimdiv, C=0.01):
        """ Initialize the classifier.
        
        Arguments:
            feature    feature to use for feature maps. Should be an instance
                       of features.Feature.
            mindimdiv  the number of divisions of the minimum dimension for
                       feature map computations.
            C          logistic regression soft-margin parameter.
        """
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.C = C

    def train(self, positives, negatives):
        """ Trains the classifier given examples of a positive image to
            recognize and negative examples.
        
        Arguments:
            positives    list of positive images to recognize.
            negatives    list of negative images.
        """
        # Find out the average aspect ratio across
        # positive samples. Use that value to define
        # the feature map dimensions.
        meanar = np.mean(map(lambda pos: float(pos.shape[1]) / pos.shape[0],
                             positives))
        # Basic algebra to get the corresponding number of rows/cols
        # from the desired minimum dimension divisions.
        self.nbrowfeat = None
        self.nbcolfeat = None
        
        if meanar > 1:
            self.nbrowfeat = self.mindimdiv
            self.nbcolfeat = self.mindimdiv * meanar
        else:
            self.nbrowfeat = int(self.mindimdiv / meanar)
            self.nbcolfeat = self.mindimdiv
        
        # Warp all the training samples to flattened
        # feature maps.
        nb_samples = len(positives) + len(negatives)
        nb_features = self.nbrowfeat * self.nbcolfeat * self.feature.dimension
        X = np.empty([nb_samples, nb_features])
        y = np.empty([nb_samples])

        i = 0
        for pos in positives:
            featmap = feat.compute_featmap(pos, self.nbrowfeat, 
                                           self.nbcolfeat, self.feature)
            X[i,:] = featmap.flatten('C')
            y[i] = 1
            i += 1
        for neg in negatives:
            featmap = feat.compute_featmap(neg, self.nbrowfeat, 
                                           self.nbcolfeat, self.feature)
            X[i,:] = featmap.flatten('C')
            y[i] = 0
            i += 1
        
        # Train a logistic regression on this data.
        self.logregr = LogisticRegression(C=self.C)
        self.logregr.fit(X, y)
    
    def predict_proba(self, images):
        """ Predicts the probability of images to be classified as
            positive by our classifier.

        Arguments:
            images    list of images.
        
        Returns:
            len(images) dimensional numpy vector containing probability
            estimates for each image, in the order they were inputted in.
        """
        # Check training has been performed
        assert self.logregr != None
        # Warp the images to flattened feature maps.
        nb_samples = len(images)
        nb_features = self.nbrowfeat * self.nbcolfeat * self.feature.dimension
        X = np.empty([nb_samples, nb_features])

        for i in range(nb_samples):
            featmap = feat.compute_featmap(images[i], self.nbrowfeat, 
                                           self.nbcolfeat, self.feature)
            X[i,:] = featmap.flatten('C')
        # Run logistic regression probability estimates.
        return self.logregr.predict_proba(X)[:,1]

