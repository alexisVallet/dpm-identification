""" Multi-class training
"""
import cv2
import numpy as np
import multiprocessing as mp
import math
import sys
import os
import json
import cPickle as pickle
import RemoteException

import features as feat
import bintraining as btrain
import ioutils as io
import model

# awful, awful hack to avoid the training data to be copied
# in an extremely inefficient manner (pickling) for each
# worker thread.
global_traindata = None

cache_folder = os.path.join('data', 'dpmid-cache')

@RemoteException.showError
def runbintrain(arguments):
    """ Runs training for a single class label.
    """
    # because multiprocessing sucks balls
    label, feature, featparams, featdim, mindimdiv, nb_parts, C, verbosity, modelname = arguments

    # check the cache
    cachefilename = os.path.join(cache_folder, modelname + '_' + label)
    if os.path.isfile(cachefilename):
        return load_model(cachefilename)

    if verbosity > 0:
        print "running training for " + repr(label) + "..."
    positives = global_traindata[label]
    negatives = reduce(lambda l1, l2: l1 + l2,
                       map(lambda k: global_traindata[k],
                           [k for k in global_traindata if k != label]))
    binmodel = btrain.binary_train(positives, negatives, feature, 
                                   featparams,
                                   featdim, nb_parts, mindimdiv, C, 
                                   verbosity > 1)
    if verbosity > 0:
        print "finished training for " + repr(label)

    # cache the results
    outfile = open(cachefilename, 'w')
    pickle.dump(binmodel, outfile)
    outfile.close()

    return binmodel

def multi_train(traindata, feature, featparams, featdim,
                nb_parts=6, mindimdiv=10, C=0.01, verbosity=0, 
                nb_cores=None, modelname=''):
    """ Trains a model for multi-class classification using deformable
        parts models. In practice, trains n binary classifiers in a one vs
        all fashion in parallel.

    Arguments:
        traindata    dictionary from class labels to corresponding sets
                     of images.
        feature      feature function to use for building feature maps.
        featdim      dimensionality of the vectors returned by the feature
                     function.
        mindimdiv    number of times to divide the smallest dimension of
                     each image to build feature maps.
        C            (L)SVM soft-margin parameter.
        verbosity    set to 0 for no messages, 1 for few messages (little
                     overhead) and 2 for a lot of messages (big overhead,
                     debug only).
    """
    # Passing the whole training data as argument to runbintrain causes
    # the data to be send by pickling to the subprocesses. Which is
    # horribly, horribly inefficient. Putting it in a global variable
    # causes it to be passed with the OS process forking semantic, which
    # in the case of linux with cpython means shared address space and
    # copy on write. Which is awesome. Yet horrible.
    global global_traindata 
    global_traindata = traindata
    if nb_cores == None:
        nb_cores = mp.cpu_count()
    pool  = mp.Pool(processes=nb_cores)
    labels = [k for k in traindata]
    feature = model.Feature.bgrhistogram
    featparams = (4,4,4)
    featdim = np.prod(featparams)

    # run each batch on its own process
    arguments = map(lambda k: (k, feature, featparams, featdim, mindimdiv, nb_parts, C, verbosity, modelname),
                    labels)
    binmodels = pool.map(runbintrain, arguments)

    return model.MultiModel(binmodels)


def train_and_save(outfile, traindata, feature, featparams, featdim, 
                   nb_parts=4, mindimdiv=7, C=0.01, verbosity=0, 
                   nb_cores=None, modelname=''):
    # run training
    model = multi_train(
        traindata,
        feature,
        featparams,
        featdim,
        nb_parts=nb_parts,
        mindimdiv=mindimdiv,
        C=C,
        verbosity=verbosity,
        nb_cores=nb_cores,
        modelname=modelname
    )
    
    # write the model to a file
    outfileobj = open(outfile, 'w')
    pickle.dump(model, outfileobj)
    outfileobj.close()
    return model

def load_model(modelfile):
    fileobj = open(modelfile)
    model = pickle.load(fileobj)
    fileobj.close()
    return model

if __name__ == "__main__":
    # command line interface to training
    # requires an image folder and bb folder. Image prefixes are taken as
    # class labels. Also requires an output file for the model.
    if len(sys.argv) < 4:
        raise ValueError("Please input a folder of training images, a folder for bounding boxes, and an output folder for the trained models.")
    imgfolder = sys.argv[1]
    bbfolder = sys.argv[2]
    outfile = sys.argv[3]
    assert os.path.isdir(imgfolder)
    assert os.path.isdir(bbfolder)
    assert os.path.isdir(os.path.dirname(outfile))
    if os.path.exists(outfile):
        print "The output file already exists, and is going to be overwritten. Do you want to continue? [y/n]"
        s = raw_input('-->')
        if s == 'n':
            sys.exit()
        elif s == 'y':
            pass
        else:
            print s + " is not a valid input, aborting."
            sys.exit()
    traindata = io.load_data(imgfolder, bbfolder)
    train_and_save(outfile, traindata)
