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
                             if (not 'asahina_mikuru' in f) and
                             f.endswith('.jpg')]

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
        print "loading negative images..."
        cls.negatives = map(loadandconvert, cls.negativefiles)

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

    def test_train_root(self):
        nbbins = (4,4,4)
        root = init.train_root(self.positives, self.negatives,
                               7, feat.labhistogram(nbbins),
                               np.prod(nbbins), 0.01)
        labimg = feat.visualize_featmap(root, 
                                        feat.labhistvis(nbbins))
        bgrimg = cv2.cvtColor(labimg, cv2.COLOR_LAB2BGR)
        cv2.namedWindow("learned root", cv2.WINDOW_NORMAL)
        cv2.imshow("learned root", bgrimg)
        cv2.waitKey(0)
                                        

if __name__ == "__main__":
    unittest.main()
