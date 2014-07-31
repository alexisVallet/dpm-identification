import numpy as np
import cv2
import cPickle as pickle
import os.path

from dpm_classifier import MultiDPMClassifier
from ioutils import load_data
from features import Feature


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
nbparts = 4
classifier = MultiDPMClassifier(
    C,
    feature,
    mindimdiv,
    nbparts,
    nb_coord_iter=4,
    nb_gd_iter=50,
    learning_rate=0.01,
    verbose=False
)

trainsamples = []
trainlabels = []

for k in traindata:
    for s in traindata[k]:
        trainsamples.append(s)
        trainlabels.append(k)



classifier.train(trainsamples, trainlabels)
