""" Unit tests for the GridSearch  class.
"""
import unittest
import numpy as np

from warpclassifier import WarpClassifier
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
        nbbins = (4,4,4)
        feature = Combine(
            HoG(9,1),
            BGRHist(nbbins,0)
        )
        mindimdiv = [5, 10, 15, 20, 25, 30]
        C = [0.1]
        learning_rate = [0.001]
        classifier = WarpClassifier()
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
            feature=[feature],
            mindimdiv=mindimdiv,
            C=C,
            learning_rate=learning_rate,
            nb_iter=[100],
            use_pca=[False],
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
