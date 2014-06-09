""" Unit tests for identification.
"""
import featpyramid as pyr
import identify as idt
import numpy as np
import cv2
import unittest

def meancolor(img):
    return np.array([
        np.mean(img[:,:,0]),
        np.mean(img[:,:,1]),
        np.mean(img[:,:,2])
    ])

class TestIdentify(unittest.TestCase):
    def test_identify(self):
        img = cv2.imread('./data/images/source/asahina_mikuru_0.jpg')
        cv2.imshow("image", img)
        nbbins = (4,4,4)
        pyramid = pyr.FeatPyramid(img, lambda img: pyr.labhistogram(img,nbbins).flatten('C'), 
                                  64, mindimdiv=10)
        rows, cols = pyramid.features[1].shape[0:2]
        # Dumb filter: a block in the middle of the image
        root = pyramid.features[1][0:5,8:cols]
        resp = idt.filter_response(pyramid.features[1], root)
        cv2.namedWindow("root response", cv2.WINDOW_NORMAL)
        cv2.imshow("root response", (resp - resp.min()) / (resp.max() - resp.min()))
        cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()
