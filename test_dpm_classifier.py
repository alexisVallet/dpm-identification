""" Unit tests for BinaryDPMClassifier.
"""
import unittest
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

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
        nbparts = 4
        classifier = BinaryDPMClassifier(
            C,
            feature,
            mindimdiv,
            nbparts,
            verbose=True,
            debug=False
        )
        label = 'rei_ayanami'
        positives = self.traindata[label]
        negatives = reduce(lambda l1,l2:l1+l2,
                           [self.traindata[l] for l in self.traindata
                            if l != label])
        print "training..."
        classifier.train(positives, negatives)
        # Display the learned DPM
        print "Deformations: " + repr(classifier.dpm.deforms)
        image = classifier.dpm.partsimage(feature.visualize)
        winname = 'learned dpm'
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, image)
        cv2.waitKey(0)
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

if __name__ == "__main__":
    unittest.main()
