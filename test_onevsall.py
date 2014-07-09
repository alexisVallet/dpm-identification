""" Unit tests for one vs all classification.
"""
import unittest
import numpy as np
import cv2

from ioutils import load_data
from onevsall import OneVSAll
from partclassifier import BinaryPartClassifier
from features import Feature

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
        initclassifier = lambda: BinaryPartClassifier(
            0.1,
            Feature('bgrhist', np.prod(nbbins), nbbins),
            10,
            verbose=False,
            debug=False,
            algorithm='l-bfgs'
        )
        cachedir = 'data/dpmid-cache/test_onevall'
        onevall = OneVSAll(
            initclassifier,
            cachedir='data/dpmid-cache/',
            verbose=True
        )
        onevall.train(trainsamples, trainlabels)

if __name__ == "__main__":
    unittest.main()
