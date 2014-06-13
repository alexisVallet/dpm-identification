""" Unit tests for featpyramid.py
"""
import featpyramid as pyr
import numpy as np
import unittest
import cv2

class TestFeatPyramid(unittest.TestCase):
    def test_initialization_labhist(self):
        """ Tests whether the histograms of blocks add up to the full image histogram.
        """
        img = cv2.imread('./data/images/source/asahina_mikuru_0.jpg')
        labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(labimg,
                                  lambda img: pyr.labhistogram(img, nbbins).flatten('C')
                                  ,64)
        fullhist = pyr.labhistogram(labimg, nbbins).flatten('C')
        pyramidsum = np.sum(pyramid.features[0], (0,1))
        np.testing.assert_almost_equal(fullhist, pyramidsum)
        

if __name__ == "__main__":
    unittest.main()
