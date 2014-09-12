""" Matching functions for linear filters on feature maps.
"""
import numpy as np
import cv2

from gdt import gdt2D
from features import Feature

def match_filter(fmap, linfilter):
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
    if fmap.shape[2] > 512:
        raise ValueError("""
Cannot compute cross-correlation for feature dimension greater than 512.
This is a limitation in OpenCV. The future switch to GPU backend for cross
correlations should hopefully fix this bug.
        """)
    fmap_ = fmap.astype(np.float32)
    linfilter_ = linfilter.astype(np.float32)
    # Run cross correlation of the filter on the map.
    response = cv2.matchTemplate(
        fmap_,
        linfilter_,
        method=cv2.TM_CCORR
    )

    return response

def match_part(fmap, partfilter, anchor, deform, 
               deform_factor, debug=False):
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
    if debug:
        respmin = response.min()
        respmax = response.max()
        print "Response max: " + repr(respmax) + ", min: " + repr(respmin)
        cv2.namedWindow('response', cv2.WINDOW_NORMAL)
        cv2.imshow('response', (response - respmin) / (respmax - respmin))
        cv2.waitKey(0)
    # Run GDT to compute score taking deformations into account and 
    # optimal displacement. GDT expects deformation costs in dx, dy, 
    # dx^2, dy^2 format so we switch things around in deform accordingly.
    dy, dx, dy2, dx2 = deform
    df, args = gdt2D(np.array([dx, dy, dx2, dy2]), response,
                     scaling=deform_factor)
    if debug:
        respmin = df.min()
        respmax = df.max()
        print "DF max: " + repr(respmax) + ", min: " + repr(respmin)
        cv2.namedWindow('df', cv2.WINDOW_NORMAL)
        cv2.imshow('df', (df - respmin) / (respmax - respmin))
        cv2.waitKey(0)
    # Get the optimal position by looking up the args array
    anci, ancj = anchor
    di, dj = args[anci, ancj] - anchor
    partsize = partfilter.shape[0]

    return (fmap[anci:anci+partsize,ancj:ancj+partsize], di, dj)
