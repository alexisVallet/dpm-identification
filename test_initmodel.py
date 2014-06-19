""" Unit tests for model initialization.
"""
import initmodel as init
import unittest
import numpy as np
import cv2
import features as feat
import featpyramid as pyr
import os

class TestInitmodel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Load positive and negative samples as LAB images.
        """
        cls.imgroot = './data/images/source/'
        cls.positivefiles = [f for f in os.listdir(cls.imgroot)
                             if 'asahina_mikuru'  in f]
        cls.negativefiles = [f for f in os.listdir(cls.imgroot)
                             if not 'asahina_mikuru' in f]

        loadandconvert = lambda filename: (
            cv2.cvtColor(
                cv2.imread(
                    os.path.join(cls.imgroot, filename)
                ).astype(np.float32)/255,
                cv2.COLOR_BGR2LAB
            )
        )
        print "loading positive images..."
        cls.positives = map(loadandconvert, cls.positivefiles)

    def test_dimred(self):
        """ Tests that dimensionality reduction indeed reduces dimension and
            keeps the required variance.
        """        
        featuremaps = []
        nbbins = (4,4,4)
        
        for labimg in self.positives:
            featuremap = pyr.compute_featmap(labimg, 7, 7, feat.labhistogram(nbbins),
                                             np.prod(nbbins))
            featuremaps.append(featuremap)

        expectedvar = 0.9
        (X, var) = init.dimred(featuremaps, expectedvar)
        self.assertGreaterEqual(var, expectedvar)
        self.assertLessEqual(X.shape[1], featuremaps[0].size)

if __name__ == "__main__":
    unittest.main()
