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
        bgrimg = cv2.imread('./data/images/source/asahina_mikuru_0.jpg')
        labimg = cv2.cvtColor(bgrimg.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(labimg,
                                  lambda img: pyr.labhistogram(img, nbbins).flatten('C')
                                  ,64)
        fullhistsum = pyr.labhistogram(labimg, nbbins).sum()
        featzerosum = pyramid.features[0].sum()

        self.assertEqual(fullhistsum, featzerosum)
        self.assertEqual(fullhistsum, np.prod(bgrimg.shape[0:2]))
        

if __name__ == "__main__":
    unittest.main()
