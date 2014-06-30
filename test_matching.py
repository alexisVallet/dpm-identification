""" Unit tests for matching algorithms.
"""
import unittest
import numpy as np
import cv2

from matching import match_filter

class TestMatching(unittest.TestCase):
    def test_match_filter(self):
        """ Tests a few invariants over randomly generated
            feature maps and filters.
        """
        for i in range(100):
            frows, fcols, fdim = np.random.randint(1, high=100, size=3)
            lrows, lcols = (np.random.randint(1, frows+1),
                            np.random.randint(1, fcols+1))
            fmap = np.random.rand(frows, fcols, fdim)
            linfilter = np.random.rand(lrows, lcols, fdim)
            response = match_filter(fmap, linfilter)
            rrows, rcols = response.shape
            if not (rrows == frows and rcols == fcols):
                print "fmap shape: " + repr(fmap.shape)
                print "lfilt shape: " + repr(linfilter.shape)
                print "respshape: " + repr(response.shape)
            self.assertEqual(rrows, frows)
            self.assertEqual(rcols, fcols)

if __name__ == "__main__":
    unittest.main()
