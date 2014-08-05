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
