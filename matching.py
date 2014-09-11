""" Matching functions for linear filters on feature maps.
"""
import numpy as np
import theano
import theano.tensor as T
import cv2

from gdt import gdt2D
from features import Feature

def match_filters(fmaps_shared):
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
    filters = T.tensor4('filters')
    cross_corr_sym = T.nnet.conv2D(fmaps_shared, filters[:,:,::-1,::-1])
    cross_corr_fn = theano.function(
        [filters],
        cross_corr_sym
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
        return cross_corr_sym(filters_tensor)

def best_response_subwindow(fmap, response, anchor, deform, deform_factor):
    """ Computes the best response subwindow and associated displacement from
        the anchor, taking into account deformation costs.
    """
    # Run GDT to compute score taking deformations into account and 
    # optimal displacement. GDT expects deformation costs in dx, dy, 
    # dx^2, dy^2 format so we switch things around in deform accordingly.
    dy, dx, dy2, dx2 = deform
    df, args = gdt2D(np.array([dx, dy, dx2, dy2]), response,
                     scaling=deform_factor)
    # Get the optimal position by looking up the args array
    anci, ancj = anchor
    di, dj = args[anci, ancj] - anchor
    partsize = partfilter.shape[0]

    return (fmap[anci:anci+partsize,ancj:ancj+partsize], di, dj)
