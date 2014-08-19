""" Simple linear classifier for warped "root" feature maps over images.
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

import features as feat
from latent_mlr import MLR
import lr
from grid_search import GridSearchMixin

class BaseWarpClassifier:
    def __init__(self, feature=feat.BGRHist((4,4,4),0), mindimdiv=10, 
                 C=0.01, learning_rate=0.01, nb_iter=100, use_pca=False, 
                 verbose=False):
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.C = C
        self.learning_rate = learning_rate
        self.nb_iter = nb_iter
        self.use_pca = use_pca
        self.verbose = verbose
        self.lr = None
        self.model_featmaps = None
        self.nbrowfeat = None
        self.nbcolfeat = None

    def train(self, samples, labels):
        # Compute feature maps.
        fmaps, self.nbrowfeat, self.nbcolfeat = (
            feat.warped_fmaps_simple(
                samples, self.mindimdiv, self.feature
            )
        )

        fvecs = map(lambda f: f.flatten('C'), fmaps)

        # Run multinomial logistic regression on it.
        self.lr = MLR(
            self.C,
            nb_iter=self.nb_iter,
            learning_rate=self.learning_rate,
            verbose=self.verbose
        )
        self.lr.train(fvecs, labels)

        # Store the learned "feature map" for each class in its proper 
        # shape.
        self.model_featmaps = []

        for i in range(self.lr.coef_.shape[1]):
            self.model_featmaps.append(
                self.lr.coef_[:,i].reshape(
                    (self.nbrowfeat, self.nbcolfeat, self.feature.dimension)
                )
            )

    def predict_proba(self, samples):
        # Compute a data matrix without dimensionality reduction.
        fmaps = map(
            lambda s: self.feature.compute_featmap(
                s, self.nbrowfeat, self.nbcolfeat
            ).flatten('C'),
            samples
        )

        return self.lr.predict_proba(fmaps)

    def predict(self, samples):
        fmaps = map(
            lambda s: self.feature.compute_featmap(
                s, self.nbrowfeat, self.nbcolfeat
            ).flatten('C'),
            samples
        )

        return self.lr.predict(fmaps)

class WarpClassifier(BaseWarpClassifier, GridSearchMixin):
    def _train_gs(self, shflsamples, shfllabels, k, **args):
        """ Override of the normal gs procedure to avoid unnecessary
            recomputation of feature maps.
        """
        best_err_rate = np.inf
        best_params = {}
        # Iterate over the feature parameters:
        for feature in args['feature']:
            for mindimdiv in args['mindimdiv']:
                # Compute feature maps.
                fmaps, self.nbrowfeat, self.nbcolfeat = (
                    feat.warped_fmaps_simple(
                        shflsamples, mindimdiv, feature
                    )
                )
                fvecs = map(lambda f: f.flatten('C'), fmaps)
                # Run GS on the MLR.
                lr = MLR()
                err_rate, params = lr._train_gs(
                    fvecs, 
                    shfllabels, 
                    k,
                    C=args['C'],
                    nb_iter=args['nb_iter'],
                    learning_rate=args['learning_rate'],
                    verbose=args['verbose']
                )
                if err_rate < best_err_rate:
                    best_params = params.copy()
                    best_params['feature'] = feature,
                    best_params['mindimdiv'] = mindimdiv
        return (best_params, best_err_rate)
