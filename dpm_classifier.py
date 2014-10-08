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
from features import max_energy_subwindow, warped_fmaps_simple, warped_fmaps_dimred, Combine, BGRHist, HoG, compute_pyramids
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

# Another hack to get around pickling restrictions.
_batch_match = None

def _best_matches(beta, pyramids, labels, args):
    global _batch_match
    dpm_size = args['size']
    deform_factor = args['df']
    cst_deform = args['cst_deform']
    max_dims = args['max_dims']
    nb_samples = len(pyramids[0])
    assert len(pyramids) == len(max_dims)
    nb_scales = len(max_dims)
    # Compile the batch matching function if it hasn't already
    # been done. One matching function for each scale
    if _batch_match == None:
        _batch_match = []
        for fmaps in pyramids:
            _batch_match.append(compile_batch_match(fmaps))
    nb_features, nb_classes = beta.shape

    # Concatenate all the filters from all the DPMs into one big list
    # for batch cross-correlation.
    dpms = []
    filters = []
    for i in range(nb_classes):
        dpm = vectortodpm(beta[:, i], dpm_size)
        dpms.append(dpm)
        filters += dpm.parts

    # Run batch cross-correlation for each individual scale.
    responses_per_scale = []
    for i in range(nb_scales):
        responses_per_scale.append(_batch_match[i](filters))

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
    max_scale = np.max(max_dims)
    
    for i in range(nb_samples):
        for j in range(nb_classes):
            subwins_and_disps = []
            for k in range(nb_parts):
                max_score = None
                max_score_subwin_disp = None
                # Get the subwindows and displacements for the best matching scale.
                for s in range(nb_scales):
                    # Compute new anchors and deformation scaling for the current scale.
                    response = responses_per_scale[s][i,j*nb_parts + k,:,:]
                    scale_factor = float(max_dims[s]) / max_scale
                    anci, ancj = np.floor(scale_factor * dpms[j].anchors[k]).astype(np.int32)
                    frows, fcols = response.shape[0:2]
                    # Clamp the scaled anchor just in case.
                    anchors = np.array([max(0, min(frows-1, anci)),
                                        max(0, min(fcols-1, ancj))])
                    # Compute the subwindow and corresponding displacement.
                    (score, subwin, di, dj) = best_response_subwin(
                        response,
                        pyramids[s][i],
                        anchors,
                        dpms[j].deforms[k] if cst_deform == None else cst_deform,
                        partsize,
                        deform_factor * scale_factor,
                        debug=False
                    )
                    # Displacements need to be scaled back to original scale.
                    scaled_disp = np.round(
                        np.array([di, dj]).astype(np.float64) / scale_factor
                    ).astype(np.int32)
                    # Keep track of the best score.
                    if max_score == None or score > max_score:
                        max_score = score
                        max_score_subwin_disp = (subwin, scaled_disp[0], scaled_disp[1])
                subwins_and_disps.append(max_score_subwin_disp)
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
                 max_dims=[5,7,9,11], nbparts=4, deform_factor=1.,
                 nb_gd_iter=50, learning_rate=0.001,
                 inc_rate=1.2, dec_rate=0.5, cst_deform=None, use_pca=None, 
                 verbose=False):
        self.C = C
        self.feature = feature
        self.max_dims = max_dims
        self.nbparts = nbparts
        self.deform_factor = deform_factor
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.nb_gd_iter = nb_gd_iter
        self.learning_rate = learning_rate
        self.cst_deform = cst_deform
        self.use_pca = use_pca
        self.verbose = verbose
        self.pca = None

    def train(self, samples, labels):
        """ Training procedure which takes precomputed feature maps as inputs.
            For efficiency purposes in grid search.
        """
        # Make sure we recompile the matching function.
        global _batch_match
        _batch_match = None
        # Initialize the model with a warping classifier, taking
        # high energy subwindows as parts. Scale is the rounded
        # mean scale across all scales of feature pyramids.
        warp_max_dim = int(round(np.mean(self.max_dims)))
        warp = WarpClassifier(
            self.feature,
            warp_max_dim,
            C=self.C,
            nb_iter=self.nb_gd_iter,
            learning_rate=self.learning_rate,
            inc_rate=self.inc_rate,
            dec_rate=self.dec_rate,
            use_pca=None,
            verbose=self.verbose
        )
        warp.train(samples, labels)
        warpmaps = warp.model_featmaps

        nb_classes = len(warpmaps)
        initdpms = []
        self.partsize = int(np.min(self.max_dims))
        
        for i in range(nb_classes):
            initdpms.append(
                _init_dpm(
                    warpmaps[i],
                    self.nbparts,
                    self.partsize
                )
            )

        # Prepare feature pyramids.
        pyramids = compute_pyramids(samples, self.max_dims, self.feature)
        nb_samples = len(samples)

        # Train the DPMs using latent MLR
        dpmsize = initdpms[0].size() # All DPMs should have the same size.
        nb_features_dpm = (dpmsize.vectorsize() if self.cst_deform == None
                           else dpmsize.vectorsize_nodeform())
        initmodel = np.empty([nb_features_dpm, nb_classes],
                             dtype=theano.config.floatX)
        
        for i in range(nb_classes):
            initmodel[:,i] = (initdpms[i].tovector() if self.cst_deform == None
                              else initdpms[i].tovector_nodeform())
        square_lead_dim = np.max(map(lambda pyrs: np.max(pyrs[0].shape[0:2]), pyramids))**2
        args = {
            'size': dpmsize,
            'cst_deform': self.cst_deform,
            'df': 1. / square_lead_dim,
            'max_dims': self.max_dims
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
            nb_samples=nb_samples,
            verbose=self.verbose
        )
        self.lmlr.train(pyramids, labels)
        # Save the DPMs for visualization purposes.
        self.dpms = []

        for i in range(self.lmlr.coef_.shape[1]):
            self.dpms.append(
                vectortodpm(self.lmlr.coef_[:,i], dpmsize)
            )

    def test_pyramids(self, samples):
        nb_samples = len(samples)
        return compute_pyramids(samples, self.max_dims, self.feature)

    def predict_proba(self, samples):
        return self.lmlr.predict_proba(self.test_pyramids(samples))

    def predict(self, samples):
        return self.lmlr.predict(self.test_pyramids(samples))

class DPMClassifier(BaseDPMClassifier, ClassifierMixin):
    pass
