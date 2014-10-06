""" Matching functions for linear filters on feature maps.
"""
import numpy as np
import cv2

from gdt import gdt2D
from features import Feature
import theano
import theano.tensor as T

def compile_batch_match(fmaps):
    """ Compiles and returns a GPU batch matching function for a dataset
        of feature maps held in a theano shared variable.

    Arguments:
        fmaps
            list of feature maps to store into the shared variable.
    Returns:
        a python function taking as input a list of filters, applying them
        to all feature maps exhaustively, returning a nb_fmaps by nb_filters
        by resp_rows by resp_cols numpy array containing all the filter responses.
    """
    nb_fmaps = len(fmaps)
    f_rows, f_cols, f_dim = fmaps[0].shape
    
    # Put the feature maps into a shared variable.
    fmaps_tensor = np.empty(
        [nb_fmaps, f_dim, f_rows, f_cols],
        dtype=theano.config.floatX
    )
    for i in range(nb_fmaps):
        for j in range(f_dim):
            fmaps_tensor[i,j] = fmaps[i][:,:,j]
    # Compile the theano function.
    fmaps_shared = theano.shared(fmaps_tensor, 'fmaps')
    filters = T.tensor4('filters')
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_fft_valid', 'conv_fft_full')
    cross_corr_sym = T.nnet.conv2d(fmaps_shared, filters[:,:,::-1,::-1])
    cross_corr_fn = theano.function(
        [filters],
        cross_corr_sym,
        mode=mode
    )
    def _helper(filters):
        nb_filters = len(filters)
        fi_rows, fi_cols, fi_dim = filters[0].shape
        assert fi_dim == f_dim
        # Convert the list of filters to the appropriate Theano data structure.
        filters_tensor = np.empty(
            [nb_filters, fi_dim, fi_rows, fi_cols],
            dtype=theano.config.floatX
        )
        for i in range(nb_filters):
            for j in range(fi_dim):
                filters_tensor[i,j] = filters[i][:,:,j]
        return cross_corr_fn(filters_tensor)
    return _helper

def _compile_crosscorr():
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_fft_valid', 'conv_fft_full')
    fmaps = T.tensor4('fmaps')
    filters = T.tensor4('filters')
    cross_corr_sym = T.nnet.conv2d(fmaps, filters[:,:,::-1,::-1])
    cross_corr_fn = theano.function(
        [fmaps, filters],
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

def best_response_subwin(response, fmap, anchor, deform, partsize, deform_factor, debug=False):
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
    # Scale the response to a nice numerical range, so the GDT doesn't run
    # into stability issues. Scale up the deformation cost accordingly, so
    # the position of the max doesn't (theoretically) change.
    up_scaling = 255. / response.max()
    gdt, args = gdt2D(np.array([dx, dy, dx2, dy2]), -up_scaling * response,
                      scaling=up_scaling * deform_factor)
    df = -gdt
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

    return (fmap[anci+di:anci+di+partsize,ancj+dj:ancj+dj+partsize], di, dj)
