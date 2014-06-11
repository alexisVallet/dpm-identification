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

    def tovector(self):
        """ Computes a vector representation for the DPM suitable for LSVM
            classification.
        
        Returns:
            A vector concatenating all the information of the DPM, for LSVM
            classification. It is simply the concatenation of the flattened
            filters, the deformation coefficients and the bias. The anchor
            positions are not included, and implicitely used in the latent
            vectors.
        """
        flattenedfilters = map(lambda f: f.flatten('C'), 
                               [self.root] + self.parts)
        toconcat = flattenedfilters + self.deforms + [np.array([self.bias])]

        return np.concatenate(toconcat)

    def size(self):
        """ Returns a DPMSize object describing the size of the DPM object,
            i.e. the information which should be unchanged by training the model.
        """
        return DPMSize(self.root.shape, map(lambda part: part.shape, self.parts),
                       self.anchors)

    def __eq__(self, other):
        allpairsclose = (lambda l1, l2: 
                         reduce(lambda b, p: b and np.allclose(p[0], p[1]), 
                                zip(l1, l2), True))

        return (np.allclose(self.root, other.root) 
                and allpairsclose(self.parts, other.parts)
                and allpairsclose(self.anchors, other.anchors)
                and allpairsclose(self.deforms, other.deforms)
                and self.bias == other.bias)

    def __repr__(self):
        return repr(self.size())

class DPMSize:
    """ Class describing the information of a DPM which are unchanged by
        training: number of parts, filter sizes and anchor positions.
    """
    def __init__(self, rootshape, partshapes, anchors):
        self.rootshape = rootshape
        self.partshapes = partshapes
        self.anchors = anchors

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

class Mixture:
    def __init__(self, dpms):
        """ Initializes a mixture of deformable parts models.

        Arguments:
            dpms    list of DPM objects.
        """
        self.dpms = dpms

    def size(self):
        """ Returns the size of the model, i.e. the information unchanged by
            training the model.

        Return:
            A MixtureSize object describing the size of this mixture.
        """

        return MixtureSize(map(lambda dpm: dpm.size(), self.dpms))

    def tovector(self):
        # special case when we have no components
        if len(self.dpms) == 0:
            return np.array([])
        
        return np.concatenate(map(lambda comp: comp.tovector(), self.dpms))

    def __eq__(self, other):

        return reduce(lambda b1, b2: b1 and b2,
                      map(lambda comps: comps[0] == comps[1],
                          zip(self.dpms, other.dpms)), True)

    def __repr__(self):
        return repr(self.size())

class MixtureSize:
    """ Describes the information of a mixture model that is unchanged by training,
        namely the size and number of each component.
    """
    def __init__(self, dpmsizes):
        self.dpmsizes = dpmsizes

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

def vectortodpm(vector, dpmsize):
    """ Compute a deformable parts model from a model vector.
    
    Arguments:
        vector  model vector to compute the DPM from.
        dpmsize DPMSize object specifying the dimensions of each filter.
    """
    # Reshape the filters one after another
    filters = []
    offset = 0

    for filtershape in [dpmsize.rootshape] + dpmsize.partshapes:
        filtersize = np.prod(filtershape)
        subvec = vector[offset:offset+filtersize]
        filters.append(subvec.reshape(filtershape, order='C'))
        offset = offset + filtersize
    
    deforms = []
    # Then just take the deformation coefficients 4 by 4
    while offset < vector.size - 1:
        deforms.append(vector[offset:offset+4])
        offset = offset + 4
    # and the bias is the last element
    bias = vector[vector.size-1]

    # build the final DPM
    return DPM(filters[0], filters[1:], dpmsize.anchors, deforms, bias)

def vectortomixture(vector, mixturesize):
    """ Converts a model vector to the corresponding mixture model.
    """
    offset = 0
    dpms = []

    for dpmsize in mixturesize.dpmsizes:
        # The size of the subvector is the sum of filter sizes, deformation
        # sizes and the bias
        subvecsize = (np.prod(dpmsize.rootshape) + 
                      np.sum(map(lambda ps: np.prod(ps),dpmsize.partshapes)) +
                      len(dpmsize.partshapes) * 4 +
                      1)
                      
        dpms.append(vectortodpm(vector[offset:offset+subvecsize],
                                dpmsize))
        offset = offset + subvecsize
    return Mixture(dpms)
