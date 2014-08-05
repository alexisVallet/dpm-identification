""" Unit tests for the WarpClassifier class.
"""
import numpy as np
import cv2
import unittest
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from ioutils import load_data
from warpclassifier import WarpClassifier
from features import Feature

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
        nbbins = (4,4,4)
        feature = Feature('bgrhist', np.prod(nbbins), nbbins)
        mindimdiv = 7
        classifier = WarpClassifier(feature, 7, verbose=True, lrimpl='llr')
        label = 'asuka_langley'
        negatives = reduce(lambda l1,l2:l1+l2,
                           [self.traindata[l] for l in self.traindata
                            if l != label])
        print "training classifier..."
        classifier.train(self.traindata[label], negatives)
        cv2.namedWindow('learned fmap', cv2.WINDOW_NORMAL)
        cv2.imshow('learned fmap', feature.vis_featmap(classifier.model_featmap))
        cv2.waitKey(0)
        print "predicting..."
        labels = []
        testimages = []

        for l in self.traindata:
            for image in self.traindata[l]:
                labels.append(l)
                testimages.append(image)

        probas = classifier.predict_proba(testimages)
        print "computing ROC curve..."
        binlabels = np.empty([len(testimages)], np.int32)
        for i in range(len(labels)):
            binlabels[i] = 1 if labels[i] == label else 0
        fpr, tpr, thresh = roc_curve(binlabels, probas)
        print "AUC = " + repr(roc_auc_score(binlabels, probas))
        plt.plot(fpr, tpr)
        plt.show()

if __name__ == "__main__":
    unittest.main()
