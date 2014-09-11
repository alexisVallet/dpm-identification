""" Multiple, deformable parts classifier.
"""
import numpy as np
import cv2
import theano
from itertools import product

from matching import match_part
from dpm import DPM, vectortodpm
from warpclassifier import WarpClassifier
from latent_mlr import LatentMLR
from features import max_energy_subwindow, warped_fmaps_simple, Combine, BGRHist, HoG
from classifier import ClassifierMixin
from dataset_transform import random_windows_fmaps

def _init_dpm(warpmap, nbparts, partsize):
    """ Initializes a DPM by greedily taking high energy subwindows
        in a warped feature map.
    """
    initparts = []
    initanchors = []
    initdeforms = [np.array([0,0,0.01,0.01])] * nbparts
    warpcopy = np.array(warpmap, copy=True)
    
    for i in range(nbparts):
        # Take the highest energy subwindow as part, zero it out
        # in the feature map, and start again.
        (maxsubwin, (ai, aj)) = max_energy_subwindow(
            warpcopy, 
            partsize
        )
        initparts.append(warpmap[ai:ai+partsize,aj:aj+partsize])
        initanchors.append(np.array([ai, aj]))
        warpcopy[ai:ai+partsize,aj:aj+partsize] = 0
    return DPM(initparts, initanchors, initdeforms)

def _best_match(dpm, featmap, deform_factor, debug=False):
    """ Computes the best matching subwindows and corresponding
        displacements of a root less deformable parts model on a
        feature map.

    Arguments:
        dpm     deformable part model to match on the feature map.
        featmap feature map to match the DPM on.

    Returns:
        (subwins, displacements) where:
        - subwins is a list of k subwindows of the (possibly zero-padded)
          input feature map, where k is the number is the number of parts
          of the DPM. Corresponds to the best possible positioning of each
          part.
        - displacements is a list of k (di, dj) pairs corresponding to the
          vertical (di) and horizonal (dj) of each optimal part position
          compared to its anchor position.
    """
    winsanddisp = map(
        lambda i: match_part(
            featmap,
            dpm.parts[i], 
            dpm.anchors[i],
            dpm.deforms[i],
            deform_factor,
            debug
        ),
        range(len(dpm.parts))
    )
    subwins = [res[0] for res in winsanddisp]
    displacements = [res[1:3] for res in winsanddisp]
    scaled_disp = []

    for i in range(len(dpm.parts)):
        di, dj = displacements[i]
        scaled_disp.append(
            (float(di) * deform_factor, float(dj) * deform_factor)
        )

    return (subwins, scaled_disp)

def _best_match_wrapper(modelvector, featmap, args):
    """ Wrapper to _best_match to convert everything into the proper
        vector format.
    """
    modelsize = args['size']
    deform_factor = args['df']

    # Compute the best match on the converted model data structure.
    (subwins, displacements) = _best_match(
        vectortodpm(modelvector, modelsize),
        featmap,
        deform_factor
    )

    # Put the computed latent values into a proper latent vector.
    latvec = np.empty([modelvector.size])
    offset = 0
    # Flatten subwindows.
    for subwin in subwins:
        flatwin = subwin.flatten('C')
        latvec[offset:offset+flatwin.size] = flatwin
        offset += flatwin.size
    # Introduce the part displacements.
    for disp in displacements:
        di, dj = disp
        latvec[offset:offset+4] = -np.array([di, dj, di**2, dj**2])
        offset = offset+4
    assert offset == modelvector.size

    return latvec

# Ugly hack.
_match_filters = None

def _best_matches(beta, fmaps_shared, labels, args):
    nb_features, nb_classes = beta.shape
    nb_samples = args['nb_samples']
    fmaps = args['fmaps']
    latents = np.empty([nb_samples, nb_features], dtype=theano.config.floatX)

    # Compute all filter responses on the GPU.
        
    for i in range(nb_samples):
        latvec = _best_match_wrapper(beta[:,labels[i]], fmaps, args)
        latents[i] = latvec
    return latents

class BaseDPMClassifier:
    """ Multi-class DPM classifier based on latent multinomial
        logistic regression.
    """
    def __init__(self, C=0.1, feature=Combine(BGRHist((4,4,4),0),HoG(9,1)), 
                 mindimdiv=10, nbparts=4, deform_factor=1.,
                 nb_coord_iter=4, nb_gd_iter=25, learning_rate=0.001,
                 inc_rate=1.2, dec_rate=0.5, nb_subwins=20, use_pca=None, 
                 verbose=False):
        self.C = C
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.nbparts = nbparts
        self.deform_factor = deform_factor
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.nb_coord_iter = nb_coord_iter
        self.nb_gd_iter = nb_gd_iter
        self.learning_rate = learning_rate
        self.nb_subwins = nb_subwins
        self.use_pca = use_pca
        self.verbose = verbose

    def _train(self, fmaps, labels, valid_fmaps=[], valid_labels=None):
        """ Training procedure which takes precomputed feature maps as inputs.
            For efficiency purposes in grid search.
        """
        global _match_filters # Urrrrrr
        # Initialize the model with a warping classifier, taking
        # high energy subwindows as parts.
        warp = WarpClassifier(
            self.feature,
            self.mindimdiv,
            C=self.C,
            nb_iter=self.nb_coord_iter*self.nb_gd_iter,
            learning_rate=self.learning_rate,
            inc_rate=self.inc_rate,
            dec_rate=self.dec_rate,
            verbose=self.verbose
        )
        warp._train(fmaps, labels, valid_fmaps, valid_labels)
        warpmaps = warp.model_featmaps

        nb_classes = len(warpmaps)
        initdpms = []
        self.partsize = self.mindimdiv // 2

        for i in range(nb_classes):
            initdpms.append(
                _init_dpm(warpmaps[i], self.nbparts, self.partsize)
            )

        nb_samples = len(fmaps)
        nb_features = self.nbrowfeat * self.nbcolfeat * self.featdim

        # Train the DPMs using latent MLR
        dpmsize = initdpms[0].size() # All DPMs should have the same size.
        nb_features_dpm = dpmsize.vectorsize()
        initmodel = np.empty([nb_features_dpm, nb_classes],
                             dtype=theano.config.floatX)
        
        for i in range(nb_classes):
            initmodel[:,i] = initdpms[i].tovector()

        # Set the deformation factor to the user supplied value, scaled
        # by 1 over the square leading feature map dimension to avoid
        # feature scaling issues in the gradient descent.
        square_lead_dim = np.max(fmaps[0].shape[0:2])**2
        args = {
            'nb_samples': nb_samples,
            'fmaps': fmaps,
            'size': dpmsize,
            'df': float(self.deform_factor) / square_lead_dim
        }

        self.lmlr = LatentMLR(
            self.C,
            _best_matches,
            args,
            initmodel,
            nb_samples=nb_samples,
            nb_coord_iter=self.nb_coord_iter,
            nb_gd_iter=self.nb_gd_iter,
            learning_rate=self.learning_rate,
            inc_rate=self.inc_rate,
            dec_rate=self.dec_rate,
            verbose=self.verbose
        )

        # Put the feature maps into a shared theano 4D tensor for more efficient
        # GPU training.
        fmaps_tensor = np.empty(
            [nb_samples, self.featdim, self.nbrowfeat, self.nbcolfeat],
            theano.config.floatX
        )
        for i in range(nb_samples):
            # The format theano expects for feature maps is annoying, so there is a bit of
            # copying going on.
            for j in range(self.featdim):
                fmaps_tensor[i,j] = fmaps[i][:,:,j]
        fmaps_shared = theano.shared(fmaps_tensor, 'fmaps')
        _match_filters = match_filters(fmaps_shared)
            
        self.lmlr.train(fmaps_shared, labels, valid_fmaps, valid_labels)
        # Save the DPMs for visualization purposes.
        self.dpms = []

        for i in range(self.lmlr.coef_.shape[1]):
            self.dpms.append(
                vectortodpm(self.lmlr.coef_[:,i], dpmsize)
            )

    def train(self, samples, labels, valid_samples=[], valid_labels=None):
        # Compute feature maps.
        if self.use_pca != None:
            fmaps, newlabels, self.pca = random_windows_fmaps(
                samples,
                labels,
                self.mindimdiv,
                self.nb_subwins,
                self.feature,
                size=0.7,
                pca=self.use_pca
            )
            self.nbrowfeat, self.nbcolfeat, self.featdim = fmaps[0].shape
            valid_fmaps = []
            if valid_labels != None or valid_samples == []:
                valid_fmaps = self.test_fmaps(valid_samples)
            self._train(fmaps, newlabels, valid_fmaps, valid_labels)
        else:
            fmaps, newlabels = random_windows_fmaps(
                samples,
                labels,
                self.mindimdiv,
                self.nb_subwins,
                self.feature,
                size=0.7,
                pca=None
            )
            self.nbrowfeat, self.nbcolfeat, self.featdim = fmaps[0].shape
            valid_fmaps = []
            if valid_labels != None or valid_samples == []:
                valid_fmaps = self.test_fmaps(valid_samples)
            self._train(fmaps, newlabels, valid_fmaps, valid_labels)

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
        return self.lmlr.predict_proba(self.test_fmaps(samples))

    def predict(self, samples):
        return self.lmlr.predict(self.test_fmaps(samples))

class DPMClassifier(BaseDPMClassifier, ClassifierMixin):
    pass
