""" Multiple, deformable parts classifier.
"""
import numpy as np
import cv2
import theano
from itertools import product

from matching import compile_match_filters, best_response_subwindow
from dpm import DPM, vectortodpm
from warpclassifier import WarpClassifier
from latent_mlr import LatentMLR
from features import max_energy_subwindow, warped_fmaps_dimred, warped_fmaps_simple, Combine, BGRHist, HoG
from classifier import ClassifierMixin

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

# Ugly hack.
_match_filters = None
_debug = False

def _best_matches(beta, fmaps_shared, labels, args):
    global _match_filters
    global _debug
    nb_features, nb_classes = beta.shape
    nb_samples = args['nb_samples']
    fmaps = args['fmaps']
    dpm_size = args['size']
    deform_factor = args['df']
    # Compile the matching function if it hasn't already been done.
    if _match_filters == None:
        _match_filters = compile_match_filters(fmaps_shared)
    # Compute the cross correlation of all filters across
    # all DPMs on each training sample in one pass on the GPU.
    # For that, need to compile the filters in one big list.
    dpms = []
    filters = []
    for i in range(nb_classes):
        dpm = vectortodpm(beta[:,i], dpm_size)
        dpms.append(dpm)
        filters += dpm.parts
    # responses_tensor is a 4D tensor of shape
    # (nb_samples, nb_filters, response_row, response_col) which
    # contains all the filter responses, in order.
    responses_tensor = _match_filters(filters)
    nb_parts = len(dpms[0].parts)
    partsize = dpms[0].parts[0].shape[0]
    # Then, from these responses, compute the best subwindows of
    # the corresponding feature maps and corresponding displacement
    # using the GDT.
    subwins_per_sample_per_class = []
    for i in range(nb_samples):
        subwins_per_class = []
        for j in range(nb_classes):
            subwins = []
            for k in range(nb_parts):
                (subwin, di, dj) = best_response_subwindow(
                    fmaps[i],
                    responses_tensor[i,j*nb_parts+k],
                    dpms[j].anchors[k],
                    dpms[j].deforms[k],
                    partsize,
                    deform_factor,
                    debug=_debug
                )
                subwins.append((subwin, di, dj))
                
            subwins_per_class.append(subwins)
        subwins_per_sample_per_class.append(subwins_per_class)
    # Now, we need to convert these subwindows into proper latent
    # vectors, and slap them into a (nb_classes, nb_samples, nb_features)
    # shaped tensor so latent MLR can properly work with it.
    latents = np.empty(
        [nb_classes, nb_samples, nb_features],
        theano.config.floatX
    )

    for i in range(nb_samples):
        featmap = fmaps[i]
        for j in range(nb_classes):
            # Put the computed latent values into a proper latent vector.
            latvec = np.empty([nb_features])
            offset = 0
            # Get the subwindows and displacements.
            subwins_and_disp = subwins_per_sample_per_class[i][j]
            subwins = [sdsp[0] for sdsp in subwins_and_disp]
            displacements = [sdsp[1:3] for sdsp in subwins_and_disp]
            # Add feature scaling for displacements.
            scaled_disp = map(
                lambda d: (float(d[0]) * deform_factor, float(d[1]) * deform_factor),
                displacements
            )
            # Flatten subwindows.
            for subwin in subwins:
                flatwin = subwin.flatten('C')
                latvec[offset:offset+flatwin.size] = flatwin
                offset += flatwin.size
            # Introduce the part displacements.
            for disp in scaled_disp:
                di, dj = disp
                latvec[offset:offset+4] = -np.array([di, dj, di**2, dj**2])
                offset = offset+4
            assert offset == nb_features
            latents[j,i] = latvec
    return (latents, np.zeros([nb_classes, nb_samples], theano.config.floatX))

class BaseDPMClassifier:
    """ Multi-class DPM classifier based on latent multinomial
        logistic regression.
    """
    def __init__(self, C=0.1, feature=Combine(BGRHist((4,4,4),0),HoG(9,1)), 
                 mindimdiv=10, nbparts=4, nb_gd_iter=100, learning_rate=0.001,
                 inc_rate=1.2, dec_rate=0.5, use_pca=None, verbose=False):
        self.C = C
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.nbparts = nbparts
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.nb_gd_iter = nb_gd_iter
        self.learning_rate = learning_rate
        self.use_pca = use_pca
        self.verbose = verbose

    def _train(self, fmaps, labels, valid_fmaps=[], valid_labels=None):
        """ Training procedure which takes precomputed feature maps as inputs.
            For efficiency purposes in grid search.
        """
        global _match_filters # Urrrrrr
        _match_filters = None
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
            'nb_classes': nb_classes,
            'fmaps': fmaps,
            'size': dpmsize,
            'df': 1. / square_lead_dim
        }

        self.lmlr = LatentMLR(
            self.C,
            _best_matches,
            args,
            initmodel,
            nb_samples=nb_samples,
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
        global _debug
        _debug = True
        probas = self.lmlr.predict_proba(self.test_fmaps(samples))
        _debug = False
        return probas

    def predict(self, samples):
        return self.lmlr.predict(self.test_fmaps(samples))

class DPMClassifier(BaseDPMClassifier, ClassifierMixin):
    pass
