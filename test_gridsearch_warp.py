""" Unit tests for the GridSearch  class.
"""
import unittest
import numpy as np

from warpclassifier import WarpClassifier
from dpm_classifier import DPMClassifier
from cross_validation import CVClassifier
from ioutils import load_data
from features import Combine, BGRHist, HoG

class TestGridSearch(unittest.TestCase):
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

    def test_classifier(self):
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
        classifier = CVClassifier(
            WarpClassifier,
            k=4,
            verbose=True,
            args={
                'feature': [Combine(HoG(9,1), BGRHist((4,4,4), 0))],
                'mindimdiv': [10],
                'C': [0.1],
                'learning_rate': [0.001],
                'nb_iter': [25, 50],
                'inc_rate': [1.2],
                'dec_rate': [0.5],
                'verbose': [True],
                'use_pca': [0.9]
            }
        )         
        classifier.train_named(trainsamples, trainlabels)

        testsamples = []
        expected = []

        for k in self.testdata:
            for s in self.testdata[k]:
                testsamples.append(s)
                expected.append(k)

        print "top-1 to to top-20 accuracy:"
        print classifier.top_accuracy_named(testsamples, expected)

if __name__ == "__main__":
    unittest.main()
