""" Unit tests for the matching algorithm.
"""
import unittest
import cv2
import numpy as np
import dpm
import features as feat
import featpyramid as pyr
import testdata as tdt
import matching as matching

class TestMatching(unittest.TestCase):
    def test_filter_response(self):
        """ Tests the result of filter response against a naive algorithm.
        """
        img = cv2.cvtColor(
            cv2.imread("data/images/source/asahina_mikuru_0.jpg").astype(np.float32)/255,
            cv2.COLOR_BGR2LAB
        )
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(img, feat.labhistogram(nbbins), np.prod(nbbins),
                                  mindimdiv = 20)
        filt = pyramid.features[1][5:16,5:16]
        actual = matching.filter_response(pyramid.features[1], filt)
        # naive computation of the expected result
        rows, cols = pyramid.features[1].shape[0:2]
        halfsize = filt.shape[0] // 2
        # padding with 0
        padded = np.zeros([rows + 2 * halfsize, cols + 2 * halfsize,
                           pyramid.features[1].shape[2]])
        padded[halfsize:halfsize+rows,halfsize:halfsize+cols] = pyramid.features[1]
        expected = np.empty([rows, cols])
        
        for i in range(0,rows):
            for j in range(0,cols):
                subwindow = padded[i:i+halfsize*2+1,j:j+halfsize*2+1]
                expected[i,j] = np.dot(filt.flatten('C'), subwindow.flatten('C'))

        cv2.namedWindow("actual", cv2.WINDOW_NORMAL)
        cv2.namedWindow("expected", cv2.WINDOW_NORMAL)
        cv2.imshow("actual", (actual - actual.min()) / (actual.max() - actual.min()))
        cv2.imshow("expected", 
                   (expected - expected.min()) / (expected.max() - expected.min()))
        cv2.waitKey(0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_dpm_matching(self):
        """ Test single component matching a test image with a random dpm
        """
        img = cv2.cvtColor(
            cv2.imread("data/images/source/asahina_mikuru_0.jpg").astype(np.float32)/255,
            cv2.COLOR_BGR2LAB
        )
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(img, feat.labhistogram(nbbins), np.prod(nbbins),
                                  mindimdiv = 20)
        # take test models out of the pyramid
        root1 = pyramid.features[1]
        part1_1 = pyramid.features[0][10:20,10:20]
        part2_1 = pyramid.features[0][20:30,20:30]
        model1 = dpm.DPM(root1, [part1_1, part2_1],
                         [np.array([10,10], np.int32), np.array([20,20], np.int32)],
                         [np.array([0,0,0.1,0.1])] * 2, 1)
        root2 = pyramid.features[1][5:15,5:15]
        part1_2 = pyramid.features[0][10:15,10:15]
        part2_2 = pyramid.features[0][15:30,15:30]
        model2 = dpm.DPM(root2, [part1_2, part2_2],
                         [np.array([0,0], np.int32), np.array([10,10], np.int32)],
                         [np.array([0,0,0.1,0.1])] * 2, 1)
        mixture = dpm.Mixture([model1,model2])
        (score, c, latvec) = matching.mixture_matching(pyramid, mixture)
        # Check that the latent vector gives the proper score
        modelvector = mixture.dpms[c].tovector()
        # Right now, the assertion does not usually hold. However, it is
        # close enough that the score is a good enough approximation for now.
        # Should come back to this bug if I have some time. Probably, there is
        # an off by one somewhere in the matching code, or a rounding error in
        # the indices for optimal displacements. Fuck it.
        scoredot = np.vdot(latvec, modelvector)
        print repr(score - scoredot)
        print repr(img.size)
        print repr(latvec.size)
        print repr(np.vdot(latvec[latvec.size-9:], modelvector[latvec.size-9:]))
        self.assertAlmostEqual(score, scoredot)

if __name__ == "__main__":
    unittest.main()