""" Matching functions for linear filters on feature maps.
"""
import numpy as np
import theano
import theano.tensor as T
import cv2

from gdt import gdt2D
from features import Feature

def compile_match_filters(fmaps_shared):
    """ Returns the response maps of a set of linear filters on a set of feature maps.
        Curried for fmaps_shared, as it involves compiling a theano function which is
        expensive.
    
    Arguments:
        fmap_shared      nb_samples by n by m by f shared theano 4D tensor.
        filter           list of n' by m' by f 3 dimensional numpy array, where
                         n' <= n and m' <= m.
        return_center    set to True if you also want the function to return
                         the coordinates of the center of the linear filter.
    Returns:
        n by m 2 dimensional numpy array, where each pixel corresponds
        to the response of the filter when its center is positioned on
        this pixel.
    """
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_gemm')
    filters = T.tensor4('filters')
    cross_corr_sym = T.nnet.conv2d(fmaps_shared, filters[:,:,::-1,::-1])
    cross_corr_fn = theano.function(
        [filters],
        cross_corr_sym,
        mode=mode
    )
    def _helper(filters_list):
        # Convert filters into a proper 4D tensor for usage with theano.
        row, col, dim = filters_list[0].shape
        filters_tensor = np.empty(
            [len(filters_list), dim, row, col],
            theano.config.floatX
        )
        
        for i in range(len(filters_list)):
            for j in range(dim):
                filters_tensor[i,j] = filters_list[i][:,:,j]
        return cross_corr_fn(filters_tensor)
    return _helper

def best_response_subwindow(fmap, response, anchor, deform, partsize, deform_factor, debug=False):
    """ Computes the best response subwindow and associated displacement from
        the anchor, taking into account deformation costs.
    """
    # Run GDT to compute score taking deformations into account and 
    # optimal displacement. GDT expects deformation costs in dx, dy, 
    # dx^2, dy^2 format so we switch things around in deform accordingly.
    dy, dx, dy2, dx2 = deform
    # GDT is numerically unstable for small response values. So we simply
    # always scale it linearly (and deformation coefficients accordingly)
    # so its standard deviation is roughly 100. This doesn't change the
    # location of the max.
    resp_scale = 100. / response.std()
    gdt, args = gdt2D(resp_scale*np.array([dx, dy, dx2, dy2], np.float32), -resp_scale*response)
    df = -gdt
    if debug:
        cv2.namedWindow('resp', cv2.WINDOW_NORMAL)
        cv2.namedWindow('gdt', cv2.WINDOW_NORMAL)
        cv2.imshow('resp', (response - response.min()) / (response.max() - response.min()))
        cv2.imshow('gdt', (df - df.min()) / (df.max() - df.min()))
        cv2.waitKey(0)
    # Get the optimal position by taking the max. Screw the args array.
    anci, ancj = anchor
    maxi, maxj = np.unravel_index(np.argmax(df), df.shape)
    di = maxi - anci
    dj = maxj - ancj

    return (fmap[anci+di:anci+di+partsize,ancj+dj:ancj+dj+partsize], di, dj)
