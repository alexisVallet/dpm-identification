""" Simple linear classifier for warped "root" feature maps over images.
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

import features as feat
from latent_lr import BinaryLR
import lr

class WarpClassifier:
    def __init__(self, feature, mindimdiv, C=0.01, learning_rate=0.01, 
                 nbiter=1000, batch_size=5, verbose=False, lrimpl='sklearn'):
        """ Initialize the classifier.
        
        Arguments:
            feature    feature to use for feature maps. Should be an instance
                       of features.Feature.
            mindimdiv  the number of divisions of the minimum dimension for
                       feature map computations.
            C          logistic regression soft-margin parameter.
        """
        assert lrimpl in ['llr', 'sklearn', 'theano']
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.C = C
        self.verbose = verbose
        self.lrimpl = lrimpl
        self.learning_rate=learning_rate
        self.nbiter = nbiter
        self.batch_size = batch_size

    def train(self, positives, negatives):
        """ Trains the classifier given examples of a positive image to
            recognize and negative examples.
        
        Arguments:
            positives    list of positive images to recognize.
            negatives    list of negative images.
        """
        # Warp all the training samples to flattened
        # feature maps.
        posmaps, negmaps, self.nbrowfeat, self.nbcolfeat = feat.warped_fmaps(
            positives, negatives, self.mindimdiv, self.feature
        )
        nb_samples = len(positives) + len(negatives)
        nb_features = self.nbrowfeat * self.nbcolfeat * self.feature.dimension
        X = np.empty([nb_samples, nb_features])
        y = np.empty([nb_samples])
        i = 0

        for pos in posmaps:
            X[i,:] = pos.flatten('C')
            y[i] = 1
            i += 1
        for neg in negmaps:
            X[i,:] = neg.flatten('C')
            y[i] = -1
            i += 1
        
        if self.lrimpl == 'llr':
            self.logregr = BinaryLR(self.C, verbose=self.verbose)
            self.logregr.fit(X, y, nb_iter=self.nbiter, learning_rate=self.learning_rate)
        elif self.lrimpl == 'theano':
            self.logregr = lr.LogisticRegression(
                self.C, 
                verbose=self.verbose)
            self.logregr.fit(X, y, nb_iter=self.nbiter, learning_rate=self.learning_rate, batch_size=self.batch_size)
        elif self.lrimpl == 'sklearn':
            self.logregr = LogisticRegression(
                penalty='l2',
                C=self.C,
                fit_intercept=True,
                dual=False
            )
            self.logregr.fit(X, y)
        # Save the (well shaped) feature map infered by logistic regression.
        self.model_featmap = self.logregr.coef_.reshape(
            (self.nbrowfeat, self.nbcolfeat, self.feature.dimension)
        )

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
        if self.lrimpl in ['llr', 'theano']:
            return self.logregr.predict_proba(X)
        elif self.lrimpl == 'sklearn':
            return self.logregr.predict_proba(X)[:,1]
