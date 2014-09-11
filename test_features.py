""" Unit tests for features.py
"""
import unittest
import cv2
import numpy as np
import features as feat

class TestFeatures(unittest.TestCase):
    def test_hog(self):
        image = cv2.imread('data/images/source/asahina_mikuru_0.jpg')
        hog = feat.HoG(9, 1)
        fmap = hog.compute_featmap(image, 20, 20)
        cv2.imshow('hog', hog.visualize_featmap(fmap))
        cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()
        
                                          
