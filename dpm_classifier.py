""" Multiple, deformable parts classifier.
"""
import numpy as np
import cv2

from matching import match_part
from dpm import DPM, vectortodpm
from warpclassifier import WarpClassifier
from latent_lr import BinaryLLR
from features import max_energy_subwindow, compute_regular_featmap

def _best_match(dpm, featmap):
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
            dpm.deforms[i]
        ),
        range(len(dpm.parts))
    )
    subwins = [res[0] for res in winsanddisp]
    displacements = [res[1:3] for res in winsanddisp]

    return (subwins, displacements)

def _best_match_wrapper(modelvector, featmap, args):
    """ Wrapper to _best_match to convert everything into the proper
        vector format.
    """
    modelsize = args['modelsize']

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
        return compute_regular_featmap(image, self.feature, self.mindimdiv)

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
        warp = WarpClassifier(self.feature, self.mindimdiv, self.C)
        warp.train(positives, negatives)
        warpmap = np.array(warp.model_featmap, copy=True)

        if self.debug:
            cv2.namedWindow('warped', cv2.WINDOW_NORMAL)
            cv2.imshow('warped', self.feature.vis_featmap(warpmap))
            cv2.waitKey(0)

        partsize = self.mindimdiv // 2
        initparts = []
        initanchors = []
        initdeforms = []
        warpcopy = np.array(warpmap, copy=True)
        
        for i in range(self.nbparts):
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
        initdpm = DPM(initparts, initanchors, initdeforms)

        if self.debug:
            cv2.namedWindow('allparts', cv2.WINDOW_NORMAL)
            cv2.imshow(
                'allparts', 
                initdpm.partsimage(self.feature.visualize)
            )
            cv2.waitKey(0)
        # Compute feature maps for all samples.
        posmaps = map(self.tofeatmap, positives)
        negmaps = map(self.tofeatmap, negatives)
        # Train the DPM using binary LLR.
        modelsize = initdpm.size()
        self.llr = BinaryLLR(
            _best_match_wrapper,
            C=self.C, 
            latent_args={'modelsize': modelsize},
            verbose=self.verbose,
            algorithm='ssm'
        )
        self.llr.fit(posmaps, negmaps, initdpm.tovector())
        # For vizualisation, compute the trained DPM
        self.dpm = vectortodpm(self.llr.model[1:], modelsize)

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
