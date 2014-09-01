""" Unit tests for the fast grid search training procedure of the DPM
    classifier.
"""
import unittest
import numpy as np
import cv2
import cPickle as pickle
import os.path

from dpm_classifier import DPMClassifier
from ioutils import load_data
from features import Combine, BGRHist, HoG

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

    def test_fast_gs(self):
        nbbins = [(3,3,3), (4,4,4), (5,5,5)]
        nb_orient = [5, 10, 20]
        feature = [Combine(
            BGRHist(nbb, 0),
            HoG(nbo, 1)
        ) for nbb in nbbins for nbo in nb_orient]
        mindimdiv = [5, 10, 20]
        C = [10., 1., 0.1, 0.01]
        nbparts = [2, 4, 8]
        deform_factor = [1.]
        learning_rate=[0.01, 0.001, 0.0001]
        classifier = DPMClassifier(verbose=True)

        trainsamples = []
        trainlabels = []

        for k in self.traindata:
            for s in self.traindata[k]:
                trainsamples.append(s)
                trainlabels.append(k)

        classifier.train_gs_fast_named(
            trainsamples,
            trainlabels,
            3,
            C=C,
            feature=feature,
            mindimdiv=mindimdiv,
            nbparts=nbparts,
            deform_factor=deform_factor,
            learning_rate=learning_rate,
            nb_iter=[100],
            nb_coord_iter=[4],
            nb_gd_iter=[25]
        )

if __name__=="__main__":
    unittest.main()
