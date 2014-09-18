""" Simple linear classifier for warped "root" feature maps over images.
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import theano

import features as feat
from latent_mlr import MLR
import lr
from classifier import ClassifierMixin
from features import warped_fmaps_simple, warped_fmaps_dimred

class BaseWarpClassifier:
    def __init__(self, feature=feat.BGRHist((4,4,4),0), mindimdiv=10, 
                 C=0.01, learning_rate=0.001, nb_iter=100, inc_rate=1.2, 
                 dec_rate=0.5, use_pca=False, verbose=False):
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.C = C
        self.learning_rate = learning_rate
        self.nb_iter = nb_iter
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.use_pca = use_pca
        self.verbose = verbose
        self.lr = None
        self.model_featmaps = None
        self.featdim = None
        self.nbrowfeat = None
        self.nbcolfeat = None

    def _train(self, fmaps, labels, valid_fmaps=[], valid_labels=None):
        """ Training procedure which takes precomputed feature maps as
            inputs. For efficiency purposes in grid search, and in the
            DPM classifier.
        """
        # Ugly hack to make things work with the DPM classifier.
        if None in (self.nbrowfeat, self.nbcolfeat, self.featdim):
            self.nbrowfeat, self.nbcolfeat, self.featdim = fmaps[0].shape
        fvecs = map(lambda f: f.flatten('C'), fmaps)
        valid_fvecs = map(lambda f: f.flatten('C'), valid_fmaps)

        # Run multinomial logistic regression on it.
        self.lr = MLR(
            self.C,
            nb_iter=self.nb_iter,
            learning_rate=self.learning_rate,
            inc_rate=self.inc_rate,
            dec_rate=self.dec_rate,
            verbose=self.verbose
        )
        self.lr.train(fvecs, labels, valid_fvecs, valid_labels)

        # Store the learned "feature map" for each class in its proper 
        # shape.
        self.model_featmaps = []

        for i in range(self.lr.coef_.shape[1]):
            fmap = self.lr.coef_[:,i].reshape(
                [self.nbrowfeat, self.nbcolfeat, self.featdim]
            )

            self.model_featmaps.append(fmap)

    def train(self, samples, labels, valid_samples=[], valid_labels=None):
        # Compute feature maps.
        if self.use_pca != None:
            fmaps, self.nbrowfeat, self.nbcolfeat, self.pca = warped_fmaps_dimred(
                samples,
                self.mindimdiv,
                self.feature,
                min_var=self.use_pca
            )
            self.featdim = fmaps[0].shape[2]
            valid_fmaps = []
            if valid_labels != None or valid_samples == []:
                valid_fmaps = self.test_fmaps(valid_samples)
            self._train(fmaps, labels, valid_fmaps, valid_labels)
        else:
            fmaps, self.nbrowfeat, self.nbcolfeat = warped_fmaps_simple(
                samples,
                self.mindimdiv,
                self.feature
            )
            self.pca = None
            self.featdim = fmaps[0].shape[2]
            valid_fmaps = []
            if valid_labels != None or valid_samples == []:
                valid_fmaps = self.test_fmaps(valid_samples)
            self._train(fmaps, labels, valid_fmaps, valid_samples)

    def test_fmaps(self, samples):
        nb_samples = len(samples)
        if self.pca != None:
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
            # X is now a data matrix of feature maps, I want just a data mat
            # of features to project.
            X_feat = X.reshape(
                [nb_samples * self.nbrowfeat * self.nbcolfeat, 
                 self.feature.dimension]
            )
            # Project the features to the principal subspace.
            X_feat_new = self.pca.transform(X_feat)
            # Convert back to feature maps.
            X_new = X_feat_new.reshape(
                [nb_samples, self.nbrowfeat, self.nbcolfeat,
                 self.pca.n_components]
            )
            # Convert it back to a feature maps representation.
            fmaps = []
            for i in range(nb_samples):
                fmaps.append(X_new[i])
            return fmaps
        else:
            fmaps = []
            for sample in samples:
                fmaps.append(self.feature.compute_featmap(
                    sample, self.nbrowfeat, self.nbcolfeat
                ))
            return fmaps

    def predict_proba(self, samples):
        return self.lr.predict_proba(map(
            lambda f: f.flatten('C'),
            self.test_fmaps(samples)
        ))

    def predict(self, samples):
        return self.lr.predict(map(
            lambda f: f.flatten('C'),
            self.test_fmaps(samples)
        ))

class WarpClassifier(BaseWarpClassifier, ClassifierMixin):
    def predict_averaged_named(self, samples):
        int_labels = self.predict(samples)
        return map(
            lambda i: self.int_to_label[i],
            self.predict_averaged(samples).tolist()
        )
