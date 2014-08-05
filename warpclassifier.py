""" Simple linear classifier for warped "root" feature maps over images.
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

import features as feat
from latent_lr import BinaryLR
from latent_mlr import MLR
import lr

class MultiWarpClassifier:
    def __init__(self, feature, mindimdiv, C=0.01, learning_rate=0.01,
                 nb_iter=100, lrimpl='sklearn', use_pca=False, 
                 verbose=False):
        assert lrimpl in ['llr', 'sklearn']
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.C = C
        self.learning_rate = learning_rate
        self.nb_iter = nb_iter
        self.lrimpl = lrimpl
        self.use_pca = use_pca
        self.verbose = verbose

    def train(self, samples, labels):
        # Compute feature maps, performing dimensionality reduction on
        # features while at it.
        fmaps = None
        if self.use_pca:
            fmaps, self.nbrowfeat, self.nbcolfeat, pca = (
                feat.warped_fmaps_dimred(
                    samples, self.mindimdiv, self.feature
                )
            )
            self.pca = pca
            self.featdim = fmaps[0].shape[2]
        else:
            fmaps, self.nbrowfeat, self.nbcolfeat = (
                feat.warped_fmaps_simple(
                    samples, self.mindimdiv, self.feature
                )
            )
            self.featdim = self.feature.dimension
        if self.verbose:
            print "Reduced feature dimension to " + repr(self.featdim)
        nb_samples = len(samples)
        nb_features = self.nbrowfeat * self.nbcolfeat * self.featdim
        self.labels_set = list(set(labels))
        label_to_int = {}

        for i in range(len(self.labels_set)):
            label_to_int[self.labels_set[i]] = i

        X = np.empty([nb_samples, nb_features])
        y = np.empty([nb_samples], dtype=np.int32)
        
        # Compute feature maps and int labels.
        for i in range(nb_samples):
            X[i] = fmaps[i].flatten('C')
            y[i] = label_to_int[labels[i]]


        if self.lrimpl == 'llr':
            self.lr = MLR(
                self.C,
                nb_iter=self.nb_iter,
                learning_rate=self.learning_rate,
                verbose=self.verbose
            )
            self.lr.fit(X, y)
        elif self.lrimpl == 'sklearn':
            self.lr = LogisticRegression(
                C=self.C
            )
            self.lr.fit(X, y)

        # Store the learned "feature map" for each class in its proper 
        # shape, projected back into to the original space.
        self.model_featmaps = []

        for i in range(self.lr.coef_.shape[1]):
            self.model_featmaps.append(
                self.lr.coef_[:,i].reshape(
                    (self.nbrowfeat, self.nbcolfeat, self.featdim)
                )
            )

    def predict_proba(self, samples):
        # Compute a data matrix without dimensionality reduction.
        X = np.empty([
            len(samples),
            self.nbrowfeat * self.nbcolfeat * self.feature.dimension
        ])

        for i in range(len(samples)):
            X[i] = feat.compute_featmap(
                samples[i], self.nbrowfeat, self.nbcolfeat, self.feature
            ).flatten('C')
        # Project the test data using PCA (if used for training)
        if self.use_pca:
            X_ = np.reshape(
                self.pca.transform(
                    np.reshape(
                        X, 
                        [len(samples)*self.nbrowfeat*self.nbcolfeat, 
                         self.feature.dimension]
                    )
                ),
                [len(samples), self.nbrowfeat*self.nbcolfeat*self.featdim]
            )
        
            return self.lr.predict_proba(X_)
        else:
            return self.lr.predict_proba(X)

    def predict(self, samples):
        X = np.empty([
            len(samples),
            self.nbrowfeat * self.nbcolfeat * self.feature.dimension
        ])

        for i in range(len(samples)):
            X[i] = feat.compute_featmap(
                samples[i], self.nbrowfeat, self.nbcolfeat, self.feature
            ).flatten('C')
        if self.use_pca:
            # Project the test data using PCA.
            X = np.reshape(
                self.pca.transform(
                    np.reshape(
                        X, 
                        [len(samples)*self.nbrowfeat*self.nbcolfeat, 
                         self.feature.dimension]
                    )
                ),
                [len(samples), self.nbrowfeat*self.nbcolfeat*self.featdim]
            )

        intlabels = self.lr.predict(X)
        labels = []

        for i in range(len(samples)):
            labels.append(self.labels_set[intlabels[i]])

        return labels

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
            self.logregr = BinaryLR(verbose=self.verbose)
            self.logregr.fit(X, y, nb_iter=self.nbiter, learning_rate=self.learning_rate, C=self.C)
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
