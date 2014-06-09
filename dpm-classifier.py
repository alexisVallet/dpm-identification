""" Implementation of an image classifier based on Felzenszwalb's deformable parts model. Intended for anime character identification.
"""
import numpy as np
import gdt
import cv2

class DPMClassifier:
    def __init__(self):
        pass

    def train(self, trainset):
        """ Trains the classifier, producing deformable models for each
            class to classify.

        Arguments:
            trainset array of triplets (I,B,l) where:
                - I is a color image
                - B = [x1,y1,x2,y2] is a bounding box in I, where (x1,y1) is the upper-left corner of the box and (x2, y2) is the down right corner. This bounding box contains the object to classify.
                - l is a class label for the object in the bounding box.
        """
        # First generate a feature pyramid for each example
        
