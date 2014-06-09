""" Deformable parts model implementation.
"""
import numpy as np

class DPM:
    def __init__(self, root, parts, anchors, deforms, bias):
        """ Initializes a deformable parts model with a root filter and a set of part
            filters with relative anchor positions.
        
        Argument:
            root    n by m by featdim numpy array for the root filter, n and m any
                    naturals.
            parts   list of k n_i by m_i by featdim numpy arrays for the parts filters,
                    n_i and m_i possibly varying for each.
            anchors list of k anchor positions for each part, as 2d (x,y) numpy arrays
                    corresponding to placement relative to the root filter in the lower
                    layer of the pyramid.
            deforms list of k 4d vectors containing deformation coefficients.
            bias    bias of the dpm in a mixture model.
        """
        assert len(parts) == len(anchors)
        assert len(parts) == len(deforms)
        self.root = root
        self.parts = parts
        self.anchors = anchors
        self.deforms = deforms
        self.bias = bias

class Mixture:
    def __init__(self, dpms):
        """ Initializes a mixture of deformable parts models.

        Arguments:
            dpms    list of DPM objects.
        """
        self.dpms = dpms
