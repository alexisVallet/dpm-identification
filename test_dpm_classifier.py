""" Unit tests for BinaryDPMClassifier.
"""
import unittest
import numpy as np

from dpm_classifier import BinaryDPMClassifier
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
        nbparts = 1
        classifier = BinaryDPMClassifier(
            C,
            feature,
            mindimdiv,
            nbparts,
            verbose=True
        )
        label = 'monkey_d_luffy'
        positives = self.traindata[label]
        negatives = reduce(lambda l1,l2:l1+l2,
                           [self.traindata[l] for l in self.traindata
                            if l != label])
        print "training..."
        classifier.train(positives, negatives)

if __name__ == "__main__":
    unittest.main()
