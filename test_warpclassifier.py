""" Unit tests for the MultiWarpClassifier class.
"""
import unittest
import numpy as np

from warpclassifier import WarpClassifier
from ioutils import load_data
from features import Combine, BGRHist, HoG

class TestWarpClassifier(unittest.TestCase):
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
            HoG(9, 1),
            BGRHist(nbbins, 0)
        )
        mindimdiv = 10
        C = 0.1
        classifier = WarpClassifier(
            feature,
            mindimdiv,
            C,
            learning_rate=0.001,
            nb_iter=100,
            verbose=True,
            use_pca=0.9
        )

        trainsamples = []
        trainlabels = []
        
        for k in self.traindata:
            for s in self.traindata[k]:
                trainsamples.append(s)
                trainlabels.append(k)

        classifier.train_named(trainsamples, trainlabels)

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