""" Unit tests for one vs all classification.
"""
import unittest
import numpy as np
import cv2

from ioutils import load_data
from onevsall import OneVSAll
from dpm_classifier import BinaryDPMClassifier
from calibration import LogisticCalibrator
from features import Feature
from warpclassifier import WarpClassifier

class TestOneVSAll(unittest.TestCase):
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

    def test_onevsall(self):
        """ Tests one vs all classification.
        """
        # Prepare training data.
        trainsamples = []
        trainlabels = []
        
        for l in self.traindata:
            for s in self.traindata[l]:
                trainsamples.append(s)
                trainlabels.append(l)
        
        # Run training.
        nbbins = (4,4,4)
        feature = Feature('bgrhist', np.prod(nbbins), nbbins)
        mindimdiv = 10
        C = 0.1
        nbparts = 1
        initclassifier = lambda: LogisticCalibrator(
            WarpClassifier(
                feature,
                mindimdiv,
                C,
                verbose=True
            ),
            verbose=True
        )
        cachedir = 'data/dpmid-cache/onevall_warp'
        onevall = OneVSAll(
            initclassifier,
            cachedir=cachedir,
            nb_cores=1,
            verbose=True
        )
        onevall.train(trainsamples, trainlabels)

        testsamples = []
        expected = []

        for k in self.testdata:
            for s in self.testdata[k]:
                testsamples.append(s)
                expected.append(k)

        predicted = onevall.predict_labels(testsamples)
        print expected
        print predicted
        correct = 0
        
        for i in range(len(predicted)):
            if predicted[i] == expected[i]:
                correct += 1

        print "Recognition rate: " + repr(float(correct) / len(predicted))

if __name__ == "__main__":
    unittest.main()
