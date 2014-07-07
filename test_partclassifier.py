""" Unit tests for the PartClassifier class.
"""
import unittest
import cv2
import numpy as np
import os
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from ioutils import load_data
from partclassifier import SinglePartClassifier
from features import Feature

class TestPartClassifier(unittest.TestCase):
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
        nbbins = (4,4,4)
        feature = Feature('bgrhist', np.prod(nbbins), nbbins)
        mindimdiv = 10
        classifier = SinglePartClassifier(
            0.1,
            feature,
            mindimdiv,
            verbose=True,
            debug=True,
            algorithm='l-bfgs'
        )
        label = 'asuka_langley'
        positives = self.traindata[label]
        negatives = reduce(lambda l1,l2:l1+l2,
                           [self.traindata[l] for l in self.traindata
                            if l != label])
        print "training..."
        classifier.train(positives, negatives)

        partimage = feature.vis_featmap(classifier.model_featmap)
        cv2.namedWindow("learned part", cv2.WINDOW_NORMAL)
        cv2.imshow("learned part", partimage)
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

        # show which parts are picked up in each test image, from highest
        # to lowest probability
        idxs = np.argsort(probas)[::-1]
        
        for i in idxs:
            print "probability: " + repr(probas[i])
            image = np.array(testsamples[i])
            (i1,j1,i2,j2) = classifier.matching_box(image)
            cv2.rectangle(image, (j1,i1), (j2,i2), (0,0,255), 4)
            cv2.imshow("image", image)
            cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()
