""" Implementation of label-preserving dataset transformation to
    increase the size of the training set artificially.
"""
import numpy as np
import cv2

from features import warped_fmaps_simple

def random_windows_fmaps(images, labels, out_mdd, win_per_sample,
                         feature, size=0.7):
    """ Generates a dataset of feature maps by taking random
        subwindows in each sample.

    Arguments:
        images
            input image samples.
        labels
            int labels in a numpy vector for each input sample.
        out_mdd
            minimum dimension of the output feature maps.
        win_per_sample
            number of new subwindow samples to generate for each sample.
        feature
            features to compute.
        overlap
            size of each window relative to the source image size.
    Returns:
        fmaps, labels where the feature maps are the new samples and
        labels are the new labels associated to these feature maps.
    """
    # Compute the warped feature maps allowing for the right output
    # minimum dimension.
    fmap_mdd = (1. + (1. - size)) * out_mdd
    fmaps, rows, cols = warped_fmaps_simple(images, fmap_mdd, feature)
    
