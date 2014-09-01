""" Implementation of label-preserving dataset transformation to
    increase the size of the training set artificially.
"""
import numpy as np
import cv2

from features import Feature, warped_fmaps_simple, warped_fmaps_dimred

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
    assert pca == None or 0. < pca <= 1.
    nb_samples = len(images)
    # Compute the warped feature maps allowing for the right output
    # minimum dimension.
    fmap_mdd = int((1. + (1. - size)) * out_mdd)
    fmaps = None
    frows = None
    fcols = None
    pca_obj = None
    if pca == None:
        fmaps, frows, fcols = warped_fmaps_simple(images, fmap_mdd, feature)
    else:
        fmaps, frows, fcols, pca_obj = warped_fmaps_dimred(
            images, fmap_mdd, feature, min_var=pca
        )
        print ("Reduced feature dimension from " + repr(feature.dimension) +
               " to " + repr(fmaps[0].shape[2]))
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

    if pca == None:
        return (out_fmaps, out_labels)
    else:
        return (out_fmaps, out_labels, pca_obj)

def left_right_flip(fmaps, labels):
    """ For each sample feature map (or image), add a left-right flipped
        version with the same label.
    """
    assert len(fmaps) == labels.size
    nb_samples = len(fmaps)
    fmaps_new = []
    labels_new = np.empty([nb_samples * 2], np.int32)

    for i in range(nb_samples):
        fmaps_new.append(fmaps[i])
        fmaps_new.append(np.fliplr(fmaps[i]))
        labels_new[2*i] = labels[i]
        labels_new[2*i+1] = labels[i]

    return (fmaps_new, labels_new)
