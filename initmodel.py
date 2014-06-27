""" Initialization procedure for a mixture of deformable parts model, as relevant for
character identification.
"""
import cv2
import numpy as np
import sklearn.svm as sklsvm
import sklearn.decomposition as skldecomp
import sklearn.cluster as sklcluster
import sklearn.metrics as skmetrics
import math

import dpm
import featpyramid as pyr
import features as feat
import training as train
import segmentation as seg

def dimred(featuremaps, minvar=0.8, verbose=False):
    """ Perform dimensionality reduction on a set of feature maps using 
    PCA.
  
    Arguments:
        featuremaps    list of feature maps to perform dimensionality 
                       reduction on
        minvar         minimum amount of variance ratio to keep.
    
    Returns:
        (X, var) where X is a n by m matrix where n is the number of 
        feature maps containing vectors representing each feature map as 
        row. var is the variance ratio preserved.
    """
    # flatten the feature maps into a data matrix
    X = np.empty([len(featuremaps), featuremaps[0].size])
    
    for i in range(0,len(featuremaps)):
        X[i,:] = featuremaps[i].flatten('C')

    # run PCA
    pca = skldecomp.PCA(n_components = minvar)

    Y = pca.fit_transform(X)

    if verbose:
        print ("PCA: " + repr(X.shape[1]) + "d to " 
               + repr(Y.shape[1]) + "d with " 
               + repr(pca.explained_variance_ratio_) + " variance kept")

    return (Y, sum(pca.explained_variance_ratio_))

def train_root(positives, negatives, mindimdiv, feature, featdim, C=0.01):
    """ Trains a root filter using a linear SVM for initialization 
        purposes.

    Arguments:
        positives positive images for the component.
        negatives negative images for the component.

    Returns:
       An initial root filter for the component.
    """
    # Initialize the root filter to the mean aspect ratio across
    # positives in the component, and size not larger than 80% of
    # the positives in the component.
    meanar = np.mean(map(lambda img: float(img.shape[1]) / float(img.shape[0]),
                         positives))
    nbcells = np.percentile(map(lambda img: np.prod(feat.regular_grid(img,mindimdiv)),
                                positives),
                            20)
    # basic linear algebra tells us the rough dimensions of the root
    nbrowfeat = int(round(math.sqrt(float(nbcells) / meanar)))
    nbcolfeat = int(round(meanar * nbrowfeat))
    # train the root filter using a linear SVM with the positives
    # in the component and all the negatives.
    tofeatmap = lambda pos: (
        feat.compute_featmap(pos, nbrowfeat, nbcolfeat, feature, featdim)
    )
    
    # prepare data for the linear SVM
    posmaps = map(tofeatmap, positives)
    negmaps = map(tofeatmap, negatives)
    roottraindata = np.empty([len(posmaps) + len(negmaps), 
                              nbrowfeat * nbcolfeat * featdim])
    roottrainlabels = np.empty([len(posmaps) + len(negmaps)], np.int32)
    i = 0
    for posmap in posmaps:
        roottraindata[i,:] = posmap.flatten('C')
        roottrainlabels[i] = 1
        i = i + 1
    for negmap in negmaps:
        roottraindata[i,:] = negmap.flatten('C')
        roottrainlabels[i] = 0
        i = i + 1
    
    # run SVM training, keep only the resulting feature weights
    roottrainer = sklsvm.LinearSVC(C=C, loss='l1')
    roottrainer.fit(roottraindata, roottrainlabels)
    featweights = roottrainer.coef_

    # The weight vector somehow lies in feature space, with some
    # strange things. Still have to figure out how to properly
    # normalize it into something useful. It looks good for color
    # histograms if we drop negative values.
    return featweights.reshape([nbrowfeat, nbcolfeat, featdim])

def cluster_comps(positives, redfeat):
    # Cluster positives with DBSCAN using correlation distance.
    # Correlation distance is very relevant here, as it is closely
    # related to the dot product - which is what we want to maximize.
    def correlation_distance(u, v):
        u_ = u - u.mean()
        v_ = v - v.mean()
        return 1 - (np.vdot(u_, v_) / (np.linalg.norm(u_) * np.linalg.norm(v_)))
    clustering = sklcluster.DBSCAN(metric=correlation_distance)
    complabels = np.round(clustering.fit_predict(redfeat)).astype(np.int32)
    mincomplabel = complabels.min()
    nbcomps = complabels.max() - mincomplabel + 1
    components = []
    
    for i in range(nbcomps):
        components.append([])
    
    # noisy samples will be labeled -1. We'll put them into their
    # own component. So if mincomplabel is negative, then we have
    # noisy samples.
    for i in range(0,len(positives)):
        label = complabels[i] - mincomplabel
        components[label].append(positives[i])

    return components

def init_parts(featmap, nb_parts):
    # set the part size to half the minimum dimension of the
    # feature map
    rows, cols = featmap.shape[0:2]
    partsize = min(rows, cols) // 2
    # compute an energy map of the feature map
    energy = np.empty([rows - partsize, cols - partsize])
    fmapcopy = np.array(featmap, copy=True)
    # then place anchors on each maximum energy point greedily
    partsandanc = []

    for p in range(nb_parts):
        # compute energy map
        for i in range(rows - partsize):
            for j in range(cols - partsize):
                subwin = fmapcopy[i:i+partsize,j:j+partsize]
                subwin[subwin < 0] = 0
                energy[i,j] = np.vdot(subwin,subwin)
        # get the max
        anci, ancj = np.unravel_index(
            np.argmax(energy),
            energy.shape
        )
        part = featmap[anci:anci+partsize,
                       ancj:ancj+partsize]
        # set the window area to 0 for next part
        fmapcopy[anci:anci+partsize,
                 ancj:ancj+partsize] = 0
        partsandanc.append((
            np.array((ancj,anci), np.int32),
            part
        ))

    return partsandanc
        

def initialize_model(positives, negatives, feature, featdim, nb_parts,
                     mindimdiv=7, C=0.01, verbose=False):
    """ Initialize a mixture model for a given class. Uses dimensionality
        reduction and clustering to guess the components. Uses 
        segmentation to guess the parts. So there is no need to specify 
        the number of either of them.

    Arguments:
        positives    positive images for the class.
        negatives    negative images for the class.
        feature      feature function to use.
        featdim      dimensionality of the features.

    Returns:
        An initial mixture model for the class. 
    """
    # compute feature maps
    featuremaps = map(lambda pos: 
                      feat.compute_featmap(
                          pos, mindimdiv, mindimdiv, 
                          feature, featdim),
                      positives)
    # dimensionality reduction
    if verbose:
        print "Running PCA..."
    (redfeat, var) = dimred(featuremaps, 0.9)
    # cluster the positives into components:
    if verbose:
        print "Clustering training samples into components.."
    comps = cluster_comps(positives, redfeat)
    if verbose:
        print "Detected " + repr(len(comps)) + " components"
    # for each cluster, compute a root
    roots = []
    i = 0
    for pos in comps:
        root = train_root(pos, negatives,
                          mindimdiv, feature, featdim)
        roots.append(root)
    # combine them into a part-less mixture, run the full LSVM
    # training algorithm on it.
    mixture = dpm.Mixture(map(lambda root: dpm.DPM(root, [], [], [], 1),
                              roots))
    mixturesize = mixture.size()
    # build feature pyramids for all samples
    def buildpyramid(img):
        return pyr.FeatPyramid(img, feature, featdim, 
                               mixturesize , mindimdiv)
    pospyr = map(buildpyramid, positives)
    negpyr = map(buildpyramid, negatives)
    # train the partless mixture
    newmixture = train.train(mixture, pospyr, negpyr, 1, C=C,
                             verbose=verbose)
    
    # initialize parts using Felzenszwalb's segmentation on linearly
    # interpolated roots
    newdpms = []
    i = 0
    for model in newmixture.dpms:
        interpolated = cv2.resize(model.root, None, fx=2, fy=2,
                                  interpolation=cv2.INTER_LINEAR)
        partsandanc = init_parts(interpolated, nb_parts)
        if verbose:
            print ("found " + repr(len(partsandanc)) 
                   + " parts for root " + repr(i))
            i = i + 1

        parts = []
        anchors = []
        for partandanc in partsandanc:
            (anc, part) = partandanc
            parts.append(part)
            anchors.append(anc)
        newdpms.append(dpm.DPM(model.root, parts, anchors,
                       map(lambda p: np.array([0,0,0.1,0.1]), parts),
                       model.bias))

    return (dpm.Mixture(newdpms), pospyr, negpyr)
