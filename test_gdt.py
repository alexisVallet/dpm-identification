import unittest
import numpy as np

import gdt
import cv2

class TestGDT(unittest.TestCase):
    def test_refimpl(self):
        img = cv2.imread('data/images/source/asahina_mikuru_0.jpg', 
                         cv2.CV_LOAD_IMAGE_GRAYSCALE)
        (dfimg1, args1) = gdt.gdt2D(np.array([0,0,1,1]), img)
        (dfimg2, args2) = gdt.gdt2D_py(np.array([0,0,1,1]), img)
        np.testing.assert_almost_equal(dfimg1, dfimg2)

    def test_args(self):
        """ Tests consistency between the distance transform and the argmax indices.
        """
        img = cv2.imread('data/images/source/asahina_mikuru_0.jpg',
                         cv2.CV_LOAD_IMAGE_GRAYSCALE)
        (dfimg, args) = gdt.gdt2D(np.array([0,0,1,1]), img)
        cv2.imshow("gdt", (dfimg - dfimg.min()) / (dfimg.max() - dfimg.min()))
        cv2.waitKey(0)
        argdt = np.empty(img.shape, dtype=np.float32)
        rows, cols = img.shape

        for i in range(0,rows):
            for j in range(0,cols):
                i1, j1 = args[i,j]
                argdt[i,j] = img[i1,j1] + (i1 - i)**2 + (j1 - j)**2
        np.testing.assert_almost_equal(dfimg, argdt)

if __name__ == "__main__":
    unittest.main()
