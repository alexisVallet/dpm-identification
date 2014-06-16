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
        print expected.shape
        print actual.shape
        print pyramid.features[1].shape
        cv2.namedWindow("actual", cv2.WINDOW_NORMAL)
        cv2.namedWindow("expected", cv2.WINDOW_NORMAL)
        cv2.imshow("actual", (actual - actual.min()) / (actual.max() - actual.min()))
        cv2.imshow("expected", (expected - expected.min()) / (expected.max() - expected.min()))
        cv2.waitKey(0)
        np.testing.assert_almost_equal(actual, expected)

    # def test_dpm_matching(self):
    #     """ Test single component matching a test image with a random dpm
    #     """
    #     img = cv2.cvtColor(
    #         cv2.imread("data/images/source/asahina_mikuru_0.jpg"),
    #         cv2.COLOR_BGR2LAB
    #     )
    #     nbbins = (4,4,4)
    #     pyramid = pyr.FeatPyramid(
    #         img,
    #         lambda img: pyr.labhistogram(img, nbbins).flatten('C'),
    #         np.prod(nbbins),
    #         mindimdiv = 20
    #     )
    #     # take a test model out of the pyramid
    #     root = pyramid.features[1]
    #     part1 = pyramid.features[0][10:20,10:20]
    #     part2 = pyramid.features[0][20:30,20:30]
    #     model = dpm.DPM(root, [part1, part2], [np.array([10,10]), np.array([20,20])],
    #                     [np.array([0,0,0.1,0.1])] * 2, 1)
    #     print repr(model)
    #     (score, pos, partresps) = matching.dpm_matching(pyramid, model)

if __name__ == "__main__":
    unittest.main()
