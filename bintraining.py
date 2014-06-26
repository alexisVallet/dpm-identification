import initmodel as init
import training as train
import model
import calibration

def binary_train(positives, negatives, feature, featparams, 
                 featdim, nb_parts, mindimdiv, C=0.01, verbose=False):
    """ Full training procedure for the binary classification
        case, including initialization and LSVM training.

    Arguments:
        positives    positive image training samples.
        negatives    negative image training samples.
        feature      feature to use for feature maps.
        featdim      dimensionality of the feature vectors.
        mindimdiv    number of cells for the minimum dimension of
                     each image.
        C            (L)SVM soft-margin parameter.
    Returns:
        A mixture model for distinguishing positives and negatives.
    """
    # initialize the mixture model, and get the feature pyramids
    # as a byproduct.
    initmixture, pospyr, negpyr = init.initialize_model(
        positives,
        negatives,
        featurefunc(feature)(featparams),
        featdim,
        nb_parts,
        mindimdiv=mindimdiv,
        C=C,
        verbose=verbose
    )
    
    # run the training procedure on the initial model
    trainedmixture = train.train(
        initmixture,
        pospyr,
        negpyr,
        nbiter=4,
        C=C,
        verbose=verbose
    )

    # train the calibrator
    calibrator = calibration.train_calibrator(
        trainedmixture,
        pospyr,
        negpyr
    )

    # return the model
    return model.BinaryModel(
        trainedmixture,
        calibrator,
        feature,
        featparams,
        featdim,
        mindimdiv
    )
