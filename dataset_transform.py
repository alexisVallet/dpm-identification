""" Implementation of label-preserving dataset transformation to
    increase the size of the training set artificially.
"""
import numpy as np
import cv2

from features import Feature, warped_fmaps_simple

def random_windows_fmaps(images, labels, out_mdd, win_per_sample,
                         feature, size=0.9, pca=None):
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
        pca
            if None, does nothing. Otherwise, should be a number between
            0 and 1 specifying the desired variance to preserve via PCA
            on the features.
    Returns:
        fmaps, labels where the feature maps are the new samples and
        labels are the new labels associated to these feature maps. If
        pca is not None, also returns the scikit learn PCA object which
        can be used for converting to and from the subspace (useful for
        visualization and prediction).
    """
    assert len(images) == len(labels)
    assert out_mdd > 0
    assert win_per_sample > 0
    assert isinstance(feature, Feature)
    assert 0. < size < 1.
    nb_samples = len(images)
    # Compute the warped feature maps allowing for the right output
    # minimum dimension.
    fmap_mdd = int((1. + (1. - size)) * out_mdd)
    fmaps, frows, fcols = warped_fmaps_simple(images, fmap_mdd, feature)
    # Compute output feature maps dimensions.
    meanar = float(fcols) / frows
    out_rows = None
    out_cols = None
    if meanar > 1:
        out_rows = out_mdd
        out_cols = out_mdd * meanar
    else:
        out_rows = int(out_mdd / meanar)
        out_cols = out_mdd
    # Construct the new dataset by picking random subwindows for each
    # feature map.
    out_fmaps = []
    out_labels = np.empty([nb_samples * win_per_sample], np.int32)
    irange = frows - out_rows
    jrange = fcols - out_cols
    offset = 0

    for i in range(nb_samples):
        for j in range(win_per_sample):
            starti = np.random.randint(irange)
            startj = np.random.randint(jrange)
            subwin = fmaps[i][starti:starti+out_rows,
                              startj:startj+out_cols]
            out_fmaps.append(subwin)
            out_labels[offset] = labels[i]
            offset += 1
    return (out_fmaps, out_labels)
