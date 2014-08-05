import unittest
import numpy as np
import cv2
import cPickle as pickle
import os.path

from dpm_classifier import MultiDPMClassifier
from ioutils import load_data
from features import Feature

class TestDPMClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "loading data..."
        # Loads the training and testdata.
        cls.testdata = load_data(
            'data/images/5-fold/0/positives/',
            'data/json/boundingboxes/'
        )
        cls.traindata = {}

        for k in range(1,5):
            folddata = load_data(
                'data/images/5-fold/' + repr(k) + '/positives/',
                'data/json/boundingboxes/'
            )
            for label in folddata:
                if not label in cls.traindata:
                    cls.traindata[label] = folddata[label]
                else:
                    cls.traindata[label] += folddata[label]

    def test_binary_dpm_classifier(self):
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
            verbose=True
        )

        trainsamples = []
        trainlabels = []

        for k in self.traindata:
            for s in self.traindata[k]:
                trainsamples.append(s)
                trainlabels.append(k)

        print "Training..."
        cachename = 'data/dpmid-cache/test_lmlr_dpm_50'
        if os.path.isfile(cachename):
            cachefile = open(cachename)
            classifier = pickle.load(cachefile)
            cachefile.close()
        else:
            classifier.train(trainsamples, trainlabels)
            cachefile = open(cachename, 'w')
            pickle.dump(classifier, cachefile)
            cachefile.close()
        print "Prediction..."

        testsamples = []
        expected = []

        for k in self.testdata:
            for s in self.testdata[k]:
                testsamples.append(s)
                expected.append(k)

        predicted = classifier.predict(testsamples)
        print expected
        print predicted
        correct = 0
        
        for i in range(len(predicted)):
            if predicted[i] == expected[i]:
                correct += 1

        print "Recognition rate: " + repr(float(correct) / len(predicted))

if __name__ == "__main__":
    unittest.main()
