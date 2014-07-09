""" Matching functions for linear filters on feature maps.
"""
import numpy as np
import cv2

from gdt import gdt2D

def match_filter(fmap, linfilter, return_padded=False):
    """ Returns the response map of a linear filter on a feature map.
        Inputs should hold 32 bits floating point coefficients for
        best performance. Will be converted if necessary.
    
    Arguments:
        fmap             n by m by f 3 dimensional numpy array.
        filter           n' by m' by f 3 dimensional numpy array, where
                         n' <= n and m' <= m.
        return_center    set to True if you also want the function to return
                         the coordinates of the center of the linear filter.
    Returns:
        n by m 2 dimensional numpy array, where each pixel corresponds
        to the response of the filter when its center is positioned on
        this pixel.
    """
    fmap_ = fmap.astype(np.float32)
    linfilter_ = linfilter.astype(np.float32)
    # Pad the feature map with zeros with half linear filter sizes.
    filtrows, filtcols = linfilter_.shape[0:2]
    hrows, hcols = (filtrows // 2, filtcols // 2)
    # opencv returns a one pixel larger map for even filters, so we
    # compensate
    roffset = 1 - filtrows % 2
    coffset = 1 - filtcols % 2
    paddedfmap = np.pad(
        fmap_,
        [(hrows,hrows-roffset),(hcols,hcols-coffset),(0,0)],
        mode='constant',
        constant_values=[(0,0)] * 3
    )
    
    # Run cross correlation of the filter on the padded map.
    response = cv2.matchTemplate(
        paddedfmap,
        linfilter_,
        method=cv2.TM_CCORR
    )
    if return_padded:
        return (response, paddedfmap)
    return response

def match_part(fmap, partfilter, anchor, deform):
    """ Matches a DPM part against a feature map.
    
    Arguments:
        fmap        feature map to match the part on.
        partfilter  linear filter of the part.
        anchor      numpy array in row,col order of the anchor point
                    of the center of the filter.
        deform      4D numpy vector of deformation coefficients.
    Returns:
        (subwin, di, dj) where:
        - subwin is the best matching subwindow from the original feature map.
        - di, dj is the displacement from the anchor position to the subwindow.
    """
    # Compute the reponse of the filter.
    response = match_filter(fmap, partfilter)
    # Pad the response before GDT computation to allow for deformations out
    # of the box. Allow a full part size out. Outside, the part doesn't match
    # anything but zeros so no point in going further.
    padding = partfilter.shape[0]
    paddedresp = np.pad(
        response,
        [(padding, padding)]*2,
        mode='constant',
        constant_values=(0,)
    )
    # Run GDT to compute score taking deformations into account and optimal
    # displacement.
    df, args = gdt2D(deform, paddedresp)
    # Get the optimal position by looking up the args array, taking into
    # account padding
    anci, ancj = anchor + np.array([padding,padding])
    di, dj = args[anci, ancj] - np.array([anci, ancj])
    # Get the corresponding subwindow from am appropriately padded feature map.
    paddedmap = np.pad(
        fmap,
        [(padding,padding)]*2 + [(0,0)],
        mode='constant',
        constant_values=(0,)
    )
    partsize = partfilter.shape[0]

    return (paddedmap[anci:anci+partsize,ancj:ancj+partsize], di, dj)
