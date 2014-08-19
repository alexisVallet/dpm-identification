""" Multiple, deformable parts classifier.
"""
import numpy as np
import cv2
import theano

from matching import match_part
from dpm import DPM, vectortodpm
from warpclassifier import WarpClassifier
from latent_mlr import LatentMLR
from features import max_energy_subwindow, warped_fmaps_simple
from grid_search import GridSearchMixin

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

def _best_matches(beta, fmaps, labels, args):
    nb_features, nb_classes = beta.shape
    nb_samples = len(fmaps)
    latents = np.empty([nb_samples, nb_features],
    dtype=theano.config.floatX)
    for i in range(nb_samples):
        latvec = _best_match_wrapper(beta[:,labels[i]], fmaps[i], args)
        latents[i] = latvec
    return latents

class BaseDPMClassifier:
    """ Multi-class DPM classifier based on latent multinomial
        logistic regression.
    """
    def __init__(self, C, feature, mindimdiv, nbparts, deform_factor=1.,
                 nb_coord_iter=4, nb_gd_iter=25, learning_rate=0.01, 
                 verbose=False):
        self.C = C
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.nbparts = nbparts
        self.deform_factor = deform_factor
        self.nb_coord_iter = nb_coord_iter
        self.nb_gd_iter = nb_gd_iter
        self.learning_rate = learning_rate
        self.verbose = verbose

    def train(self, samples, labels):
        # Initialize the model with a warping classifier, taking
        # high energy subwindows as parts.
        warp = WarpClassifier(
            self.feature,
            self.mindimdiv,
            C=self.C,
            learning_rate=self.learning_rate,
            verbose=self.verbose
        )
        warp.train(samples, labels)
        warpmaps = warp.model_featmaps

        nb_classes = len(warpmaps)        
        initdpms = []
        self.partsize = self.mindimdiv // 2

        for i in range(nb_classes):
            initdpms.append(
                _init_dpm(warpmaps[i], self.nbparts, self.partsize)
            )

        # Compute feature maps.
        fmaps, self.nbrowfeat, self.nbcolfeat = warped_fmaps_simple(
            samples, self.mindimdiv, self.feature
        )
        nb_samples = len(samples)
        nb_features = self.nbrowfeat * self.nbcolfeat * self.feature.dimension

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
            'size': dpmsize,
            'df': float(self.deform_factor) / square_lead_dim
        }

        self.lmlr = LatentMLR(
            self.C,
            _best_matches,
            args,
            initmodel,
            nb_coord_iter=self.nb_coord_iter,
            nb_gd_iter=self.nb_gd_iter,
            learning_rate=self.learning_rate,
            verbose=self.verbose
        )
        self.lmlr.train(fmaps, labels)
        # Save the DPMs for visualization purposes.
        self.dpms = []

        for i in range(self.lmlr.coef_.shape[1]):
            self.dpms.append(
                vectortodpm(self.lmlr.coef_[:,i], dpmsize)
            )
       
    def predict_proba(self, samples):
        # Convert images to feature maps.
        fmaps = map(lambda s: self.feature.compute_featmap(
            s, 
            self.nbrowfeat, 
            self.nbcolfeat
        ), samples)

        return self.lmlr.predict_proba(fmaps)

    def predict(self, samples):
        # Convert images to feature maps.
        fmaps = map(lambda s: self.feature.compute_featmap(
            s, 
            self.nbrowfeat, 
            self.nbcolfeat
        ), samples)

        return self.lmlr.predict(fmaps)

class DPMClassifier(BaseDPMClassifier, GridSearchMixin):
    pass
