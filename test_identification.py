""" Unit tests for identification.
"""
import unittest
import os
import cv2
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import numpy as np

import multitraining as mtrain
import identification as ident
import ioutils
import features as feat

class TestIdentification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Loads the model for the first fold of cross
            validation and the corresponding test data.
        """
        print "loading data..."
        # ground truth bounding box for test, not ideal but it'll do
        cls.testimages = ioutils.load_data(
            os.path.join('data', 'images', '5-fold', '0', 'positives'),
            os.path.join('data', 'json', 'boundingboxes')
        )

    def test_binary_identification(self):
        print "loading model..."
        binmodel = mtrain.load_model(
            os.path.join('data', 'dpmid-cache', 'fold_0_sakura_haruno')
        )

        images = []

        for label in self.testimages:
            for image in self.testimages[label]:
                images.append(image)

        distances = []
        for image in images:
            dist = ident.adhoc_identification(binmodel, image)
            distances.append(dist)
        
        avgdist = np.mean(distances)
        print "average distance " + repr(avgdist)

        best = np.argsort(distances)

        for i in best:
            print repr(distances[i])
            cv2.imshow('image', images[i])
            cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()
