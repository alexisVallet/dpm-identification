""" Initialization procedure for a mixture of deformable parts model, as relevant for
character identification.
"""
import dpm
import cv2
import numpy as np
import sklearn as skl

def mean_aspect_ratio(images):
    """ Computes the mean aspect ratio across a set of images.
    
    Arguments:
        images array of images.
    Return:
        floating point mean aspect ratio.
    """
    return sum(map(lambda img: img.shape[1]/img.shape[0]))

def initialize_roots(trainingimages, feature):
    """ Initialize the root filters of a mixture of deformable parts model. Uses
        a method inspired by Divvala et al., 2012 to cluster the training data, except
        we first apply PCA for dimensionality reduction and use DBSCALE in order to
        determine a suitable number of roots.
    
    Arguments:
        trainingimages  array of training images.
        feature         feature function to use, takes an image as input and should
                        return a f-dimensional vector.
    Return:
        array of roots represented by feature maps.
    """
    # Warp (resize) all examples to the average aspect ratio across all 
    # images, compute feature maps for each of them.
    meanar = mean_aspect_ratio(trainingimages)
    
