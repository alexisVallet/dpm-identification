""" Unit tests for features.py
"""
import unittest
import cv2
import numpy as np
import features as feat

class TestFeatures(unittest.TestCase):
    # def test_lab_histogram(self):
    #     bgrimg = cv2.imread('data/images/source/asahina_mikuru_0.jpg')
    #     labimg = cv2.cvtColor(bgrimg.astype(np.float32)/255, 
    #                           cv2.COLOR_BGR2LAB)
    #     nbbins1 = (3,3,3)
    #     featmap1 = feat.compute_regular_featmap(labimg, 10, 
    #                                             feat.labhistogram(nbbins1),
    #                                             np.prod(nbbins1))
    #     nbbins2 = (6,6,6)
    #     featmap2 = feat.compute_regular_featmap(labimg, 10,
    #                                             feat.labhistogram(nbbins2),
    #                                             np.prod(nbbins2))
        

    def test_visualize_featmap(self):
        nbbins = (4,4,4)
        featdim = np.prod(nbbins)
        mindimdiv = 10
        bgrimg = cv2.imread('35562572_m.png')
        feature = feat.Feature('bgrhist', featdim, nbbins)
        featmap = feat.compute_regular_featmap(bgrimg, feature, mindimdiv)
        rows, cols = featmap.shape[0:2]
        featimage = feature.vis_featmap(featmap)
        flatfeatimage = feature.vis_featmap(featmap.reshape([1, rows*cols, featdim]))
        cv2.namedWindow("feature map", cv2.WINDOW_NORMAL)
        cv2.namedWindow("flat map", cv2.WINDOW_NORMAL)
        cv2.imshow("feature map", featimage)
        cv2.imshow("flat map", flatfeatimage)
        cv2.waitKey(0)

if __name__ == "__main__":
    unittest.main()
        
                                          
