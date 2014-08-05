""" Unit tests for the GridSearch  class.
"""
import unittest
import numpy as np

from dpm_classifier import MultiDPMClassifier
from grid_search import GridSearch
from ioutils import load_data
from features import Feature

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
        feature = Feature('bgrhist', np.prod(nbbins), nbbins)
        mindimdiv = [5, 10, 20]
        C = [1, 0.1, 0.01]
        learning_rate = [0.1, 0.01, 0.001]
        nbparts = [1, 2, 4, 8]
        classifier = GridSearch(
            lambda args: MultiDPMClassifier(
                args['C'],
                feature,
                args['mdd'],
                args['nbp'],
                learning_rate=args['lr'],
                nb_coord_iter=4,
                nb_gd_iter=25,
                verbose=True
            ),{
                'mdd': mindimdiv,
                'C': C,
                'lr': learning_rate,
                'nbp': nbparts
            },
            k=3,
            verbose=True
        )

        trainsamples = []
        trainlabels = []
        
        for k in self.traindata:
            for s in self.traindata[k]:
                trainsamples.append(s)
                trainlabels.append(k)

        classifier.train(trainsamples, trainlabels)

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
