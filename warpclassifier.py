""" Simple linear classifier for warped "root" feature maps over images.
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import theano

import features as feat
from latent_mlr import MLR
import lr
from grid_search import GridSearchMixin
from dataset_transform import random_windows_fmaps, left_right_flip

class BaseWarpClassifier:
    def __init__(self, feature=feat.BGRHist((4,4,4),0), mindimdiv=10, 
                 C=0.01, opt='rprop', learning_rate=0.001, nb_iter=100, inc_rate=1.2, 
                 dec_rate=0.5, use_pca=False, verbose=False):
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.C = C
        self.opt = opt
        self.learning_rate = learning_rate
        self.nb_iter = nb_iter
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.use_pca = use_pca
        self.verbose = verbose
        self.lr = None
        self.model_featmaps = None
        self.nbrowfeat = None
        self.nbcolfeat = None

    def _train(self, fmaps, labels):
        """ Training procedure which takes precomputed feature maps as
            inputs. For efficiency purposes in grid search, and in the
            DPM classifier.
        """
        self.nbrowfeat, self.nbcolfeat, self.featdim = fmaps[0].shape
        fvecs = map(lambda f: f.flatten('C'), fmaps)

        # Run multinomial logistic regression on it.
        self.lr = MLR(
            self.C,
            nb_iter=self.nb_iter,
            opt=self.opt,
            learning_rate=self.learning_rate,
            inc_rate=self.inc_rate,
            dec_rate=self.dec_rate,
            verbose=self.verbose
        )
        self.lr.train(fvecs, labels)

        # Store the learned "feature map" for each class in its proper 
        # shape.
        self.model_featmaps = []


        for i in range(self.lr.coef_.shape[1]):
            fmap = self.lr.coef_[:,i].reshape(
                [self.nbrowfeat, self.nbcolfeat, self.featdim]
            )

            self.model_featmaps.append(fmap)

    def train(self, samples, labels):
        # Compute feature maps.
        if self.use_pca != None:
            fmaps, newlabels, self.pca = random_windows_fmaps(
                samples,
                labels,
                self.mindimdiv,
                10,
                self.feature,
                size=0.7,
                pca=self.use_pca
            )
            self._train(fmaps, newlabels)
        else:
            fmaps, newlabels = random_windows_fmaps(
                samples,
                labels,
                self.mindimdiv,
                10,
                self.feature,
                size=0.7,
                pca=None
            )
            self._train(fmaps, newlabels)

    def predict_proba(self, samples):
        nb_samples = len(samples)
        # Compute a data matrix without dimensionality reduction.
        X = np.empty(
            [nb_samples,
             self.nbrowfeat,
             self.nbcolfeat,
             self.feature.dimension],
            dtype=theano.config.floatX
        )

        for i in range(nb_samples):
            X[i] = self.feature.compute_featmap(
                samples[i], self.nbrowfeat, self.nbcolfeat
            )
        # X is now a data matrix of feature maps, I want just a data matrix
        # of features to project.
        X_feat = X.reshape(
            [nb_samples * self.nbrowfeat * self.nbcolfeat, 
             self.feature.dimension]
        )
        # Project the features to the principal subspace.
        X_feat_new = self.pca.transform(X_feat)
        # Convert back to feature maps.
        X_new = X_feat_new.reshape(
            [nb_samples, self.nbrowfeat * self.nbcolfeat *
             self.pca.n_components]
        )
        # Convert it back to a feature maps representation.
        fmaps = []
        for i in range(nb_samples):
            fmaps.append(X_new[i])

        return self.lr.predict_proba(fmaps)

    def _predict(self, fmaps):
        return self.lr.predict(fmaps)

    def predict(self, samples):
        nb_samples = len(samples)
        # Compute a data matrix without dimensionality reduction.
        X = np.empty(
            [nb_samples,
             self.nbrowfeat,
             self.nbcolfeat,
             self.feature.dimension],
            dtype=theano.config.floatX
        )

        for i in range(nb_samples):
            X[i] = self.feature.compute_featmap(
                samples[i], self.nbrowfeat, self.nbcolfeat
            )
        # X is now a data matrix of feature maps, I want just a data matrix
        # of features to project.
        X_feat = X.reshape(
            [nb_samples * self.nbrowfeat * self.nbcolfeat, 
             self.feature.dimension]
        )
        # Project the features to the principal subspace.
        X_feat_new = self.pca.transform(X_feat)
        # Convert back to feature maps.
        X_new = X_feat_new.reshape(
            [nb_samples, self.nbrowfeat * self.nbcolfeat *
             self.pca.n_components]
        )
        # Convert it back to a feature maps representation.
        fmaps = []
        for i in range(nb_samples):
            fmaps.append(X_new[i])

        return self._predict(fmaps)

class WarpClassifier(BaseWarpClassifier, GridSearchMixin):
    pass
