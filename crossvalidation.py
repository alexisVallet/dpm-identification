""" Runs cross validation for DPM identification
"""
import os
import time
import numpy as np
import cv2
import sys

import multitraining as mtrain
import ioutils as io
import model

def train_cv(k, feature, featparams, featdim, cvroot, bbfolder, 
             outfolder, verbose=True):
    """ Trains models for k-fold cross validation, given specific
        bounding boxes for training. Sequentially train each fold
        in parallel, write each model to the disk as it goes. Checks
        that the model file isn't already existing. If it is, skips
        training (i.e. we assume the existing file is correct). Also
        trains score calibrators.

    Arguments:
        k         number of folds for the cross validation.
        cvroot    root folder of the cross-validation data. Should
                  contain the precomputed folds.
        bbfolder  folder for training bounding boxes. Should be ground
                  truth in most cases.
        outfolder folder to write the resulting models to.
    """
    for fold in range(k):
        if verbose:
            print "Training for fold " + repr(fold)
        # if the model file exists, skip
        outname = os.path.join(outfolder, 'model_' + repr(fold))
        if os.path.isfile(outname):
            continue
        traindata = {}
        # get the training data from all the other folds
        for otherfold in [f for f in range(k) if f != fold]:
            foldername = os.path.join(cvroot, repr(otherfold),
                                      'positives')
            folddata = io.load_data(foldername, bbfolder)
            
            for label in folddata:
                if not label in traindata:
                    traindata[label] = folddata[label]
                else:
                    traindata[label] += folddata[label]
        if verbose:
            print "Training for labels:"
            print [label for label in traindata]
            total = 0
            for label in traindata:
                total += len(traindata[label])
                print label + " has " + repr(len(traindata[label])) + " training samples"
            print "total " + repr(total) + " samples"

        # run the multi training algorithm
        starttime = time.clock()
        mtrain.train_and_save(
            outname,
            traindata,
            feature,
            featparams,
            featdim,
            verbosity=1 if verbose else 0,
            modelname='fold_' + repr(fold)
        )
        endtime = time.clock()
        elapsed = endtime - starttime
        print ("Ran training for fold " + repr(fold) + " in " 
               + repr(elapsed) + " seconds.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError("Please input k, the CV folder, the bounding boxes folder and the output folder.")    
    k = int(sys.argv[1])
    cvroot = sys.argv[2]
    bbfolder = sys.argv[3]
    outfolder = sys.argv[4]
    assert os.path.isdir(cvroot)
    assert os.path.isdir(bbfolder)
    assert os.path.isdir(outfolder)
    feature = model.Feature.labhistogram
    featparams = (4,4,4)
    featdim = np.prod(featparams)
    train_cv(k, feature, featparams, featdim, cvroot, bbfolder, outfolder,
             verbose=True)
