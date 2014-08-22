""" Unit tests for the GridSearch  class.
"""
import unittest
import numpy as np

from warpclassifier import WarpClassifier
from dpm_classifier import DPMClassifier
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
        feature = [Combine(BGRHist((nbb,nbb,nbb),0), HoG(nbo, 1)) 
                   for nbb in [4,5] for nbo in [5,10]]
        mindimdiv = [10, 15]
        C = [0.1, 0.01]
        learning_rate = [0.01, 0.001]
        nbparts = [5,10]
        classifier = DPMClassifier()
        trainsamples = []
        trainlabels = []
        
        for k in self.traindata:
            for s in self.traindata[k]:
                trainsamples.append(s)
                trainlabels.append(k)

        classifier.train_gs_named(
            trainsamples, 
            trainlabels,
            3,
            C=C,
            feature=feature,
            mindimdiv=mindimdiv,
            nbparts=nbparts,
            learning_rate=learning_rate,
            deform_factor=[1.],
            nb_coord_iter=[4],
            nb_gd_iter=[25],
            verbose=[True]
        )

        testsamples = []
        expected = []

        for k in self.testdata:
            for s in self.testdata[k]:
                testsamples.append(s)
                expected.append(k)

        predicted = classifier.predict_named(testsamples)
        print expected
        print predicted
        correct = 0
        
        for i in range(len(predicted)):
            if predicted[i] == expected[i]:
                correct += 1

        print "Recognition rate: " + repr(float(correct) / len(predicted))

if __name__ == "__main__":
    unittest.main()
