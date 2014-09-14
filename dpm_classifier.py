""" Multiple, deformable parts classifier.
"""
import numpy as np
import cv2
import theano
from itertools import product

from matching import compile_batch_match, best_response_subwin
from dpm import DPM, vectortodpm
from warpclassifier import WarpClassifier
from latent_mlr import LatentMLR
from features import max_energy_subwindow, warped_fmaps_simple, warped_fmaps_dimred, Combine, BGRHist, HoG
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

# Another hack to get around pickling restrictions.
_batch_match = None

def _best_matches(beta, fmaps, labels, args):
    global _batch_match
    # Compile the batch matching function if it hasn't already
    # been done.
    if _batch_match == None:
        _batch_match = compile_batch_match(fmaps)
    nb_features, nb_classes = beta.shape
    nb_samples = len(fmaps)
    dpm_size = args['size']
    deform_factor = args['df']
    cst_deform = args['cst_deform']

    # Concatenate all the filters from all the DPMs into one big list
    # for batch cross-correlation.
    dpms = []
    filters = []
    for i in range(nb_classes):
        dpm = vectortodpm(beta[:, i], dpm_size)
        dpms.append(dpm)
        filters += dpm.parts

    # Run batch cross-correlation.
    responses = _batch_match(filters)

    # Compute the corresponding subwindows and displacements for each
    # part of each DPM on each sample using GDT, and store the corresponding
    # latent vectors into the 4D tensor required by latent MLR.
    nb_parts = len(dpms[0].parts)
    latents = np.empty(
        [nb_classes, nb_samples, nb_features],
        dtype=theano.config.floatX
    )
    lat_cst = np.zeros([nb_classes, nb_samples], theano.config.floatX)
    partsize = dpms[0].parts[0].shape[0]
    
    for i in range(nb_samples):
        for j in range(nb_classes):
            subwins_and_disps = []
            for k in range(nb_parts):
                # Compute the subwindow and corresponding displacement.
                subwin_and_disp = best_response_subwin(
                    responses[i,j*nb_parts + k,:,:],
                    fmaps[i],
                    dpms[j].anchors[k],
                    dpms[j].deforms[k] if cst_deform == None else cst_deform,
                    deform_factor,
                    partsize,
                    debug=False
                )
                subwins_and_disps.append(subwin_and_disp)
            # Put the computed latent values into a proper latent vector.
            latvec = np.empty([nb_features])
            offset = 0
            subwins = [sad[0] for sad in subwins_and_disps]
            # Introduce the deformation factor into the displacements.
            disps = [(float(sad[1]) * deform_factor, float(sad[2]) * deform_factor)
                     for sad in subwins_and_disps]
            
            # Flatten subwindows.
            for subwin in subwins:
                flatwin = subwin.flatten('C')
                latvec[offset:offset+flatwin.size] = flatwin
                offset += flatwin.size
            if cst_deform == None:
                # Introduce the part displacements, with deformation factor.
                for disp in disps:
                    di, dj = disp
                    latvec[offset:offset+4] = -np.array([di, dj, di**2, dj**2])
                    offset = offset+4
            else:
                # Compute the deformation cost using the constant deformation
                # coefficients.
                for disp in disps:
                    di, dj = disp
                    dx, dy, dx2, dy2 = cst_deform
                    lat_cst[j,i] -= dy*di + dx*dj + dy2*di**2 + dx2*dj**2
            assert offset == nb_features
            # Put the computed latent vector into the tensor.
            latents[j,i] = latvec
    
    return (latents, lat_cst)

class BaseDPMClassifier:
    """ Multi-class DPM classifier based on latent multinomial
        logistic regression.
    """
    def __init__(self, C=0.1, feature=Combine(BGRHist((4,4,4),0),HoG(9,1)), 
                 mindimdiv=10, nbparts=4, deform_factor=1.,
                 nb_gd_iter=50, learning_rate=0.001,
                 inc_rate=1.2, dec_rate=0.5, cst_deform=None, use_pca=None, 
                 verbose=False):
        self.C = C
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.nbparts = nbparts
        self.deform_factor = deform_factor
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.nb_gd_iter = nb_gd_iter
        self.learning_rate = learning_rate
        self.cst_deform = cst_deform
        self.use_pca = use_pca
        self.verbose = verbose

    def _train(self, fmaps, labels, valid_fmaps=[], valid_labels=None):
        """ Training procedure which takes precomputed feature maps as inputs.
            For efficiency purposes in grid search.
        """
        # Initialize the model with a warping classifier, taking
        # high energy subwindows as parts.
        warp = WarpClassifier(
            self.feature,
            self.mindimdiv,
            C=self.C,
            nb_iter=self.nb_gd_iter,
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
                _init_dpm(
                    warpmaps[i],
                    self.nbparts,
                    self.partsize
                )
            )

        nb_samples = len(fmaps)
        nb_features = self.nbrowfeat * self.nbcolfeat * self.featdim

        # Train the DPMs using latent MLR
        dpmsize = initdpms[0].size() # All DPMs should have the same size.
        nb_features_dpm = (dpmsize.vectorsize() if self.cst_deform == None
                           else dpmsize.vectorsize_nodeform())
        initmodel = np.empty([nb_features_dpm, nb_classes],
                             dtype=theano.config.floatX)
        
        for i in range(nb_classes):
            initmodel[:,i] = (initdpms[i].tovector() if self.cst_deform == None
                              else initdpms[i].tovector_nodeform())

        # Set the deformation factor to the user supplied value, scaled
        # by 1 over the square leading feature map dimension to avoid
        # feature scaling issues in the gradient descent.
        square_lead_dim = np.max(fmaps[0].shape[0:2])**2
        args = {
            'size': dpmsize,
            'df': float(self.deform_factor) / square_lead_dim,
            'cst_deform': self.cst_deform
        }

        self.lmlr = LatentMLR(
            self.C,
            _best_matches,
            args,
            initmodel,
            nb_gd_iter=self.nb_gd_iter,
            learning_rate=self.learning_rate,
            inc_rate=self.inc_rate,
            dec_rate=self.dec_rate,
            verbose=self.verbose
        )
        self.lmlr.train(fmaps, labels, valid_fmaps, valid_labels)
        # Save the DPMs for visualization purposes.
        self.dpms = []

        for i in range(self.lmlr.coef_.shape[1]):
            self.dpms.append(
                vectortodpm(self.lmlr.coef_[:,i], dpmsize)
            )

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
        return self.lmlr.predict_proba(self.test_fmaps(samples))

    def predict(self, samples):
        return self.lmlr.predict(self.test_fmaps(samples))

class DPMClassifier(BaseDPMClassifier, ClassifierMixin):
    pass
