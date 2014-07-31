""" Multiple, deformable parts classifier.
"""
import numpy as np
import cv2
import theano

from matching import match_part
from dpm import DPM, vectortodpm
from warpclassifier import WarpClassifier, MultiWarpClassifier
from latent_lr import LatentLR
from latent_mlr import LatentMLR
from features import max_energy_subwindow, compute_featmap, warped_fmaps_simple

def _init_dpm(warpmap, nbparts, partsize):
    """ Initializes a DPM by greedily taking high energy subwindows
        in a warped feature map.
    """
    initparts = []
    initanchors = []
    initdeforms = []
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
        initdeforms.append(np.array([0,0,0.1,0.1]))
        warpcopy[ai:ai+partsize,aj:aj+partsize] = 0
    return DPM(initparts, initanchors, initdeforms)

def _best_match(dpm, featmap, debug=False):
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
            debug
        ),
        range(len(dpm.parts))
    )
    subwins = [res[0] for res in winsanddisp]
    displacements = [res[1:3] for res in winsanddisp]

    # Scale displacements linearly to the [-1;1] range to keep them
    # at a scale comparable to subwindows. The squared displacements are 
    # limited to a range of leading_fmap_dim^2 in value. We'll
    # assume that all feature maps are warped to the same dimension.
    leading_dim = max(featmap.shape[0], featmap.shape[1])**2
    scaled_disp = []

    for i in range(len(dpm.parts)):
        di, dj = displacements[i]
        scaled_disp.append((float(di) / leading_dim, float(dj) / leading_dim))

    return (subwins, scaled_disp)

def _best_match_wrapper(modelvector, featmap, args):
    """ Wrapper to _best_match to convert everything into the proper
        vector format.
    """
    modelsize = args

    # Compute the best match on the converted model data structure.
    (subwins, displacements) = _best_match(
        vectortodpm(modelvector, modelsize),
        featmap
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
        if abs(di) > 30 or abs(dj) > 30:
            print "Invalid displacements: "
            print (di, dj)
        latvec[offset:offset+4] = -np.array([di, dj, di**2, dj**2])
        offset = offset+4
    assert offset == modelvector.size

    return latvec

class BinaryDPMClassifier:
    def __init__(self, C, feature, mindimdiv, nbparts, verbose=False,
                 debug=False):
        """ Initializes the classifier with a given number of parts.
        
        Arguments:
            C         Soft margin parameter for latent logistic regression.
            feature   Feature function to use for individual image patches.
            mindimdiv Number of splits along each image's smallest dimension.
            nbparts   Number of parts to train the classifier with.
            verbose   Set to True for regular information messages.
            debug     If set to True, will stop execution at various point
                      showing the current model being trained.
        """
        self.C = C
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.nbparts = nbparts
        self.verbose = verbose
        self.debug = debug

    def tofeatmap(self, image):
        return compute_featmap(
            image, self.nbrowfeat, self.nbcolfeat, self.feature
        )

    def train(self, positives, negatives):
        """ Fits the classifier given a set of positive images and a set
            of negative images.
        
        Arguments:
            positives    array of positive images, i.e. the classifier 
                         should return a probability close to 1 for them.
            negatives    array of negative images, i.e. the classifier 
                         should return a probability close to 0 for them.
        """
        # Initialize the model by training a warping classifier on all
        # images, and greedily taking square high energy subwindows as
        # parts.
        warp = WarpClassifier(self.feature, self.mindimdiv, C=self.C)
        warp.train(positives, negatives)
        warpmap = np.array(warp.model_featmap, copy=True)

        if self.debug:
            cv2.namedWindow('warped', cv2.WINDOW_NORMAL)
            cv2.imshow('warped', self.feature.vis_featmap(warpmap))
            cv2.waitKey(0)

        partsize = self.mindimdiv // 2
        
        initdpm = _init_dpm(warpmap, self.nbparts, partsize)

        if self.debug:
            cv2.namedWindow('allparts', cv2.WINDOW_NORMAL)
            cv2.imshow(
                'allparts', 
                initdpm.partsimage(self.feature.visualize)
            )
            cv2.waitKey(0)
        # Compute feature maps for all samples.
        self.nbrowfeat = warp.nbrowfeat
        self.nbcolfeat = warp.nbcolfeat
        posmaps = map(self.tofeatmap, positives)
        negmaps = map(self.tofeatmap, negatives)
        # Train the DPM using binary LLR.
        modelsize = initdpm.size()
        self.llr = LatentLR(
            _best_match_wrapper,
            latent_args={'modelsize': modelsize, 'debug': self.debug},
            verbose=self.verbose
        )
        self.llr.train(self.C, posmaps, negmaps, initdpm.tovector(), 0)
        # For vizualisation, compute the trained DPM
        self.dpm = vectortodpm(self.llr.coef_, modelsize)

    def predict_proba(self, images):
        """ Predicts probabilities that images are positive samples.

        Arguments:
            images    image to predict probabilities for.
        """
        assert self.llr != None

        fmaps = map(self.tofeatmap, images)
        
        # Use the internal latent logistic regression to predict 
        # probabilities.
        return self.llr.predict_proba(fmaps)

def _best_matches(beta, fmaps, labels, args):
    nb_features, nb_classes = beta.shape
    nb_samples = len(fmaps)
    latents = np.empty([nb_samples, nb_features], 
                       dtype=theano.config.floatX)

    for i in range(nb_samples):
        latvec = _best_match_wrapper(beta[:,labels[i]], fmaps[i], args)
        latents[i] = latvec

    return latents

class MultiDPMClassifier:
    """ Multi-class DPM classifier based on latent multinomial
        logistic regression.
    """
    def __init__(self, C, feature, mindimdiv, nbparts, nb_coord_iter=4,
                 nb_gd_iter=25, learning_rate=0.01, verbose=False):
        self.C = C
        self.feature = feature
        self.mindimdiv = mindimdiv
        self.nbparts = nbparts
        self.nb_coord_iter = nb_coord_iter
        self.nb_gd_iter = nb_gd_iter
        self.learning_rate = learning_rate
        self.verbose = verbose

    def train(self, samples, labels):
        # Initialize the model with a warping classifier, taking
        # high energy subwindows as parts.
        warp = MultiWarpClassifier(
            self.feature,
            self.mindimdiv,
            C=self.C,
            lrimpl='llr',
            verbose = self.verbose
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
        
        # Put labels in 0..k-1 range.
        self.labels_set = list(set(labels))
        label_to_int = {}

        for i in range(len(self.labels_set)):
            label_to_int[self.labels_set[i]] = i

        y = np.empty([nb_samples], dtype=np.int32)
        
        for i in range(nb_samples):
            y[i] = label_to_int[labels[i]]

        # Combine feature map and label in the samples.
        samples = fmaps

        # Train the DPMs using latent MLR
        dpmsize = initdpms[0].size() # All DPMs should have the same size.
        nb_features_dpm = dpmsize.vectorsize()
        initmodel = np.empty([nb_features_dpm, nb_classes],
                             dtype=theano.config.floatX)
        
        for i in range(nb_classes):
            initmodel[:,i] = initdpms[i].tovector()

        self.lmlr = LatentMLR(
            self.C,
            _best_matches,
            dpmsize,
            initmodel,
            nb_coord_iter=self.nb_coord_iter,
            nb_gd_iter=self.nb_gd_iter,
            learning_rate=self.learning_rate,
            verbose=self.verbose
        )
        self.lmlr.fit(samples, y)
       
    def predict_proba(self, samples):
        # Convert images to feature maps.
        fmaps = map(lambda s: compute_featmap(
            s, 
            self.nbrowfeat, 
            self.nbcolfeat, 
            self.feature
        ), samples)

        return self.lmlr.predict_proba(fmaps)

    def predict(self, samples):
        # Convert images to feature maps.
        fmaps = map(lambda s: compute_featmap(
            s, 
            self.nbrowfeat, 
            self.nbcolfeat, 
            self.feature
        ), samples)
        intlabels = self.lmlr.predict(fmaps)
        labels = []

        for i in range(len(samples)):
            labels.append(self.labels_set[intlabels[i]])

        return labels
