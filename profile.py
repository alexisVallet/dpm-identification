import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from dpm_classifier import BinaryDPMClassifier
from calibration import LogisticCalibrator
from ioutils import load_data
from features import Feature

if __name__ == "__main__":
    print "loading data..."
    # Loads the training and testdata.
    testdata = load_data(
        'data/images/5-fold/0/positives/',
        'data/json/boundingboxes/'
    )
    traindata = {}
    
    for k in range(1,5):
        folddata = load_data(
            'data/images/5-fold/' + repr(k) + '/positives/',
            'data/json/boundingboxes/'
        )
        for label in folddata:
            if not label in traindata:
                traindata[label] = folddata[label]
            else:
                traindata[label] += folddata[label]

    nbbins = (4,4,4)
    feature = Feature('bgrhist', np.prod(nbbins), nbbins)
    mindimdiv = 10
    C = 0.1
    nbparts = 1
    classifier = LogisticCalibrator(
        BinaryDPMClassifier(
            C,
            feature,
            mindimdiv,
            nbparts,
            verbose=False,
            debug=False
        ),
        verbose=False
    )
    label = 'asuka_langley'
    positives = traindata[label]
    negatives = reduce(lambda l1,l2:l1+l2,
                       [traindata[l] for l in traindata
                        if l != label])
    print "training..."
    classifier.train(positives, negatives)

