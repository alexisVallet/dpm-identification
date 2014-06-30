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
            (response, padded) = match_filter(fmap, linfilter, 
                                              return_padded=True)
            rrows, rcols = response.shape
            if not (rrows == frows and rcols == fcols):
                print "fmap shape: " + repr(fmap.shape)
                print "lfilt shape: " + repr(linfilter.shape)
                print "respshape: " + repr(response.shape)
            self.assertEqual(rrows, frows)
            self.assertEqual(rcols, fcols)
            maxi, maxj = np.unravel_index(
                np.argmax(response),
                response.shape
            )
            subwin = padded[maxi:maxi+lrows, maxj:maxj+lcols]
            # for some reason OpenCV's code is wildly inaccurate - can only get
            # accuracy up to one decimal place. This might be because OpenCV's
            # code relies on the fast convolution algorithm, which may be unstable
            # numerically.
            self.assertAlmostEqual(
                np.vdot(linfilter, subwin),
                response[maxi,maxj],
                places=1
            )

if __name__ == "__main__":
    unittest.main()
