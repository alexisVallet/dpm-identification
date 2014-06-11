""" Unit tests for the dpm module.
"""
import unittest
import dpm
import numpy as np
import numpy.random as npr
import featpyramid as pyr
import cv2
import testdata

class TestDPM(unittest.TestCase):
    def testvectorconversion(self):
        """ Tests that one can convert to and from vector and get the
            same thing.
        """
        img = cv2.imread('./data/images/source/asahina_mikuru_0.jpg')
        labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(labimg, 
                                  lambda img: pyr.labhistogram(img, nbbins).flatten('C')
                                  ,64)
        # Try out a 100 random mixtures on the pyramid
        for i in range(0,100):
            mixture = testdata.randommixture(pyramid)
            vector = mixture.tovector()
            size = mixture.size()
            nmixture = dpm.vectortomixture(vector, size)
            nvector = nmixture.tovector()
            self.assertEqual(mixture, nmixture)
            self.assertTrue(np.array_equal(vector, nvector))

if __name__ == "__main__":
    unittest.main()
