import unittest
import numpy as np
import cv2
import cPickle as pickle
import os.path

from dpm_classifier import DPMClassifier
from ioutils import load_data
from features import Combine, BGRHist, HoG

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
        feature = Combine(
            BGRHist(nbbins, 0),
            HoG(9,1)
        )
        mindimdiv = 10
        C = 0.1
        nbparts = 4
        deform_factor = 1.
        classifier = DPMClassifier(
            C,
            feature,
            mindimdiv,
            nbparts,
            deform_factor,
            nb_gd_iter=25,
            learning_rate=0.001,
            inc_rate=1.2,
            dec_rate=0.5,
            nb_subwins=1,
            use_pca=0.9,
            verbose=True
        )

        trainsamples = []
        trainlabels = []

        for k in self.traindata:
            for s in self.traindata[k]:
                trainsamples.append(s)
                trainlabels.append(k)

        print "Training..."
        classifier.train_named(trainsamples, trainlabels)

        print "Prediction..."
        testsamples = []
        expected = []

        for k in self.testdata:
            for s in self.testdata[k]:
                testsamples.append(s)
                expected.append(k)
        print "top-1 to top-20 accuracy:"
        print classifier.top_accuracy_named(testsamples, expected)

if __name__ == "__main__":
    unittest.main()
