# -*- coding: utf-8 -*-
import unittest
import numpy as np
import cv2
import cPickle as pickle
import os.path

from dpm_classifier import DPMClassifier
from ioutils import load_data, load_data_pixiv
from features import Combine, BGRHist, HoG
from cross_validation import k_fold_split

class TestDPMClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "loading data..."
        # Loads the training and testdata.
        testdata = load_data(
            'data/images/5-fold/0/positives/',
            'data/json/boundingboxes/'
        )
        traindata = {}
        for k in range(1,5):
            folddata = load_data(
                'data/images/5-fold/' + repr(k) + '/positives/',
                'data/json/boundingboxes/'
                )
            for label in folddata:
                if not label in traindata:
                    traindata[label] = folddata[label]
                else:
                    traindata[label] += folddata[label]
        cls.traindata = []
        cls.trainlabels = []

        for k in traindata:
            for s in traindata[k]:
                cls.traindata.append(s)
                cls.trainlabels.append(k)
        cls.testdata = []
        cls.testlabels = []

        for k in testdata:
            for s in testdata[k]:
                cls.testdata.append(s)
                cls.testlabels.append(k)

    def test_binary_dpm_classifier(self):
        nbbins = (4,4,4)
        feature = Combine(
            BGRHist(nbbins, 0),
            HoG(9,1)
        )
        max_dims=[10]
        C=0.1
        nbparts = 4
        deform_factor = 1.
        classifier = DPMClassifier(
            C,
            feature,
            max_dims,
            nbparts,
            deform_factor,
            nb_gd_iter=75,
            learning_rate=0.001,
            inc_rate=1.2,
            dec_rate=0.5,
            cst_deform=[0,0,10000.,10000.],
            use_pca=None,
            verbose=True
        )

        print "Training..."
        classifier.train_named(self.traindata, self.trainlabels)
        print "Top-1 to top-20 accuracy:"
        print classifier.top_accuracy_named(self.testdata, self.testlabels)

if __name__ == "__main__":
    unittest.main()
