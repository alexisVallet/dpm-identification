""" Matching functions for linear filters on feature maps.
"""
import numpy as np
import cv2

from gdt import gdt2D
from features import Feature
import theano
import theano.tensor as T

def _compile_crosscorr():
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_gemm')
    fmap = T.tensor4('fmap')
    filter_ = T.tensor4('filter')
    cross_corr_sym = T.nnet.conv2d(fmap,filter_[:,:,::-1,::-1])
    cross_corr_fn = theano.function(
        [fmap, filter_],
        cross_corr_sym,
        mode=mode
    )
    return cross_corr_fn

_cross_corr = _compile_crosscorr()

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
    f_rows, f_cols, f_dim = fmap.shape
    l_rows, l_cols, l_dim = linfilter.shape
    assert f_dim == l_dim
    assert l_rows <= f_rows and l_cols <= f_cols
    # Putting it in the theano format.
    fmap_ = np.empty(
        [1, f_dim, f_rows, f_cols],
        dtype=theano.config.floatX
    )
    linfilter_ = np.empty(
        [1, l_dim, l_rows, l_cols],
        dtype=theano.config.floatX
    )

    for i in range(f_dim):
        fmap_[0,i] = fmap[:,:,i]
        linfilter_[0,i] = linfilter[:,:,i]
    
    # Run cross correlation of the filter on the map.
    response = _cross_corr(fmap_, linfilter_)
    nb_resp, nb_filters, r_rows, r_cols = response.shape
    assert nb_resp == 1
    assert nb_filters == 1
    
    return response.reshape([r_rows, r_cols])

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
