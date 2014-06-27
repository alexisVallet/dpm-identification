""" Multi-class identification.
"""
import numpy as np
import cv2
import sklearn.linear_model as sklinear

import matching
import featpyramid as pyr
import model
import dpm
import features as feat

def binary_identification(binmodel, images, calibrate=True, parts=True):
    """ Returns the score of a binary mixture model on a
        list of images.
    """
    # if no parts specified, run the algorithm on model with no parts
    if not parts:
        newdpms = []
        for rootmodel in binmodel.mixture.dpms:
            newdpms.append(
                dpm.DPM(rootmodel.root, [], [], [], rootmodel.bias)
            )
        newmodel = model.BinaryModel(
            dpm.Mixture(newdpms),
            binmodel.calibrator,
            binmodel.feature,
            binmodel.featparams,
            binmodel.featdim,
            binmodel.mindimdiv
        )

        return binary_identification(newmodel, images, calibrate,
                                     parts=True)

    modelsize = binmodel.mixture.size()
    modelvector = binmodel.mixture.tovector()
    feature = model.featurefunc(binmodel.feature)(binmodel.featparams)

    rawscores = np.empty([len(images),1])
    i = 0
    for image in images:
        # compute the pyramid for the image
        pyramid = pyr.FeatPyramid(image, feature, 
                                  binmodel.featdim, modelsize, 
                                  binmodel.mindimdiv)
        # match the model against the pyramid
        score, c, latvec = matching.mixture_matching(pyramid, 
                                                     binmodel.mixture)
        print "latent vector min: " + repr(latvec.min())
        print "latent vector max: " + repr(latvec.max())
        print "latent vector mean: " + repr(latvec.mean())
        rawscores[i,0] = score
        i = i + 1

    print "model vector min: " + repr(modelvector.min())
    print "model vector max: " + repr(modelvector.max())
    print "model vector mean: " + repr(modelvector.mean())
    print "nb negative values: " + repr(modelvector[modelvector < 0].size)
    idxs = []
    for i in range(modelvector.size):
        if modelvector[i] < 0:
            idxs.append(i)
    print "negative indexes: " + repr(idxs)
    print "negative values: " + repr(modelvector[modelvector < 0])
    print "model vector size: " + repr(modelvector.size)
    for comp in binmodel.mixture.dpms:
        print "comp vec size: " + repr(comp.size().vectorsize())

    # calibrate the scores to the [0;1] range if specified
    if calibrate:
        return binmodel.calibrator.predict_proba(rawscores)
    else:
        return rawscores

def adhoc_identification(binmodel, image):
    rootimages = []
    feature = model.featurefunc(binmodel.feature)(binmodel.featparams)
    featdim = binmodel.featdim
    featvis = feat.bgrhistvis(binmodel.featparams)

    for rootmodel in binmodel.mixture.dpms:
        rootimages.append(
            feat.visualize_featmap(
                rootmodel.root,
                featvis,
                blocksize=(1,1)
            )
        )

    imagepyr = pyr.FeatPyramid(
        image, 
        feature, 
        featdim,
        binmodel.mixture.size(),
        binmodel.mindimdiv
    )

    distances = []

    for i in range(len(rootimages)):
        rootimage = cv2.cvtColor(rootimages[i], cv2.COLOR_BGR2LAB)
        imagerepr = feat.visualize_featmap(
            imagepyr.rootfeatures[i],
            featvis,
            blocksize=(1,1)
        )
        imagerepr = cv2.cvtColor(imagerepr, cv2.COLOR_BGR2LAB)
        
        rows, cols = rootimage.shape[0:2]
        avgpixdist = 0.0
        for i in range(rows):
            for j in range(cols):
                avgpixdist += np.linalg.norm(
                    rootimage[i,j] - imagerepr[i,j]
                )
        distances.append(avgpixdist / (rows*cols))

    return min(distances)

def multi_identification(multimodel, images):
    """ Returns the score for each class of a multimodel on
        a set of images.
    """
