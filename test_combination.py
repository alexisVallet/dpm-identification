""" Unit tests for combination.py
"""
import unittest
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from combination import Combination
from dpm_classifier import BinaryDPMClassifier
from warpclassifier import WarpClassifier
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

    def test_binary_dpm_combination(self):
        nbbins = (4,4,4)
        feature = Feature('bgrhist', np.prod(nbbins), nbbins)
        warpmindimdiv = 5
        C = 0.1
        nbparts = 2
        classifier = Combination([
            WarpClassifier(
                feature,
                warpmindimdiv,
                C,
                verbose=True
            ),
            BinaryDPMClassifier(
                C,
                feature,
                warpmindimdiv * 2,
                nbparts,
                verbose=True,
                debug=False
            )
        ])
            
        label = 'rei_ayanami'
        positives = self.traindata[label]
        negatives = reduce(lambda l1,l2:l1+l2,
                           [self.traindata[l] for l in self.traindata
                            if l != label])
        print "training..."
        classifier.train(positives, negatives)
        print "predicting..."

        testlabels = []
        testsamples = []

        for l in self.testdata:
            for t in self.testdata[l]:
                testlabels.append(l)
                testsamples.append(t)

        probas = classifier.predict_proba(testsamples)
        
        print "computing ROC curve"
        binlabels = np.array([1 if l == label else 0 for l in testlabels])
        fpr, tpr, threshs = roc_curve(binlabels, probas)
        print "auc: " + repr(roc_auc_score(binlabels, probas))
        print "thresholds:"
        print repr(threshs)
        plt.plot(fpr, tpr)
        plt.show()

        for c in classifier.classifiers:
            probas = c.predict_proba(testsamples)
            print "computing ROC curve"
            fpr, tpr, threshs = roc_curve(binlabels, probas)
            print "auc: " + repr(roc_auc_score(binlabels, probas))
            print "thresholds:"
            print repr(threshs)
            plt.plot(fpr, tpr)
            plt.show()

if __name__ == "__main__":
    unittest.main()
