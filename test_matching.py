""" Unit tests for the matching algorithm.
"""
import unittest
import cv2
import numpy as np
import dpm
import featpyramid as pyr
import testdata as tdt
import matching as matching

class TestMatching(unittest.TestCase):
    def test_filter_response(self):
        """ Tests the result of filter response against a naive algorithm.
        """
        img = cv2.cvtColor(
            cv2.imread("data/images/source/asahina_mikuru_0.jpg"),
            cv2.COLOR_BGR2LAB
        )
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(
            img,
            lambda img: pyr.labhistogram(img, nbbins).flatten('C'),
            np.prod(nbbins),
            mindimdiv = 20
        )
        filt = pyramid.features[1][5:16,5:16]
        actual = matching.filter_response(pyramid.features[1], filt)
        # naive computation of the expected result
        rows, cols = pyramid.features[1].shape[0:2]
        halfsize = filt.shape[0] // 2
        # padding with 0
        padded = np.zeros([rows + 2 * halfsize, cols + 2 * halfsize,
                           pyramid.features[1].shape[2]])
        padded[halfsize:halfsize+rows,halfsize:halfsize+cols] = (
            pyramid.features[1]
        )
        expected = np.empty([rows, cols])
        
        for i in range(0,rows):
            for j in range(0,cols):
                subwindow = padded[i:i+halfsize*2+1,j:j+halfsize*2+1]
                expected[i,j] = np.dot(filt.flatten('C'), subwindow.flatten('C'))
        np.testing.assert_almost_equal(actual, expected)

    def test_dpm_matching(self):
        """ Test single component matching a test image with a random dpm
        """
        img = cv2.cvtColor(
            cv2.imread("data/images/source/asahina_mikuru_2.jpg").astype(np.float32) / 255,
            cv2.COLOR_BGR2LAB
        )
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(
            img,
            lambda img: pyr.labhistogram(img, nbbins).flatten('C'),
            np.prod(nbbins),
            mindimdiv = 20
        )
        # take a test model out of the pyramid
        root = pyramid.features[1]
        part1 = pyramid.features[0][10:20,10:20]
        part2 = pyramid.features[0][20:30,20:30]
        model = dpm.DPM(root, [part1, part2], [np.array([10,10]), np.array([20,20])],
                        [np.array([0,0,0.1,0.1])] * 2, 1)
        (score, pos, partresps) = matching.dpm_matching(pyramid, model)
        i = 0
        for partresp in partresps:
            cv2.namedWindow("part " + repr(i), cv2.WINDOW_NORMAL)
            cv2.imshow("part " + repr(i), 
                       (partresp - partresp.min()) / (partresp.max() - partresp.min()))
            i = i + 1
        cv2.waitKey(0)

    def test_mixture_matching(self):
        img = cv2.cvtColor(
            cv2.imread("data/images/source/asahina_mikuru_2.jpg").astype(np.float32) / 255,
            cv2.COLOR_BGR2LAB
        )
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(
            img,
            lambda img: pyr.labhistogram(img, nbbins).flatten('C'),
            np.prod(nbbins),
            mindimdiv = 20
        )
        # take test models out of the pyramid
        root1 = pyramid.features[1]
        part1_1 = pyramid.features[0][10:20,10:20]
        part2_1 = pyramid.features[0][20:30,20:30]
        model1 = dpm.DPM(root1, [part1_1, part2_1],
                         [np.array([10,10]), np.array([20,20])],
                         [np.array([0,0,0.1,0.1])] * 2, 1)
        root2 = pyramid.features[1][5:15,5:15]
        part1_2 = pyramid.features[0][10:15,10:15]
        part2_2 = pyramid.features[0][15:30,15:30]
        model2 = dpm.DPM(root2, [part1_2, part2_2],
                         [np.array([0,0]), np.array([10,10])],
                         [np.array([0,0,0.1,0.1])] * 2, 1)
        mixture = dpm.Mixture([model1,model2])
        (score, c, latvec) = matching.mixture_matching(pyramid, mixture)
        # Check that the latent vector gives the proper score
        modelvector = mixture.dpms[c].tovector()
        print repr(modelvector.sum())
        print repr(latvec.sum())
        self.assertAlmostEqual(score, np.vdot(latvec, modelvector))

if __name__ == "__main__":
    unittest.main()
