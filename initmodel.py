""" Initialization procedure for a mixture of deformable parts model, as relevant for
character identification.
"""
import dpm
import featpyramid as pyr
import features as feat
import cv2
import numpy as np
import sklearn.svm as sklsvm
import sklearn.decomposition as skldecomp
import sklearn.cluster as sklcluster

def dimred(featuremaps, minvar=0.8):
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
    # train the root filter using a linear SVM with the positives
    # in the component and all the negatives.
    
    # if the aspect ratio > 1, then we want more columns, otherwise
    # we want more rows - while the minimum should stay the same.
    nbrowfeat, nbcolfeat = (None, None)
    
    if meanar > 1:
        nbrowfeat, nbcolfeat = (mindimdiv, int(round(float(mindimdiv) * meanar)))
    else:
        nbrowfeat, nbcolfeat = (int(round(float(mindimdiv) * (1/meanar))), mindimdiv)

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

    # The weight vector should be in column-major (Fortran) order.
    return featweights.reshape([nbrowfeat, nbcolfeat, featdim])

def initialize_model(positives, negatives, feature, featdim, mindimdiv=7, C=0.01):
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
    # Warp positive images into a common feature space
    featuremaps = map(lambda pos: feat.compute_featmap(pos, mindimdiv, mindimdiv, 
                                                      feature, featdim),
                      positives)
    # Run dimensionality reduction for clustering
    (redfeat, var) = dimred(featuremaps, 0.9)
    
    # Cluster them into components using DBSCAN
    clustering = sklcluster.DBSCAN(metric='correlation')
    complabels = np.round(clustering.fit_predict(redfeat)).astype(np.int32)
    mincomplabel = complabels.min()
    nbcomps = complabels.max() - mnincomplabel
    components = [[]] * nbcomps
    
    for i in range(0,len(positives)):
        components[complabels[i]].append(positives[i])
    
    # Initialize root filters for each component
    rootfilters = []
