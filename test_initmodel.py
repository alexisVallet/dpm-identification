""" Unit tests for model initialization.
"""
import unittest
import numpy as np
import cv2
import os
import json

import initmodel as init
import features as feat
import featpyramid as pyr

class TestInitmodel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # function loading the image, crop it to only the
        # bounding box of the character
        """ Load positive and negative samples as LAB images.
        """
        cls.imgroot = './data/images/source/'
        cls.bbroot = './data/json/boundingboxes/'
        cls.positivefiles = [f for f in os.listdir(cls.imgroot)
                             if 'asahina_mikuru'  in f]
        cls.negativefiles = [f for f in os.listdir(cls.imgroot)
                             if (not 'asahina_mikuru' in f) and
                             f.endswith('.jpg')]
        # only keep a random subset of 100 negative images 
        # (loading all the negatives takes too long)
        # fixed seed for repeatable results.
        np.random.seed(1)
        cls.negativefiles = np.random.permutation(
            cls.negativefiles
        )[0:100]

        def loadandconvert(filename):
            bgrimg = cv2.imread(
                os.path.join(cls.imgroot, filename)
            )
            # load the corresponding bounding boxes
            jsonfile = open(
                os.path.join(
                    cls.bbroot, 
                    os.path.splitext(filename)[0] + '_bb.json'
                )
            )
            boxes = json.load(jsonfile)
            jsonfile.close()
            # If there are multiple boxes in the image,
            # assume the first one is the right one.
            [[x1,y1],[x2,y2]] = boxes[0]
            # crop the image
            charbgrimg = bgrimg[y1:y2,x1:x2]

            return charbgrimg

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
        
        for bgrimg in self.positives:
            featuremap = feat.compute_featmap(bgrimg, 7, 7, feat.bgrhistogram(nbbins),
                                              np.prod(nbbins))
            featuremaps.append(featuremap)

        expectedvar = 0.9
        (X, var) = init.dimred(featuremaps, expectedvar)
        self.assertGreaterEqual(var, expectedvar)
        self.assertLessEqual(X.shape[1], featuremaps[0].size)

    def test_train_root(self):
        nbbins = (7,7,7)
        root = init.train_root(self.positives, self.negatives,
                               15, feat.bgrhistogram(nbbins),
                               np.prod(nbbins))
        img = feat.visualize_featmap(root,
                                     feat.bgrhistvis(nbbins),
                                     blocksize=(1,1))
        cv2.namedWindow("learned root", cv2.WINDOW_NORMAL)
        cv2.imshow("learned root", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()
