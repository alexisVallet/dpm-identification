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
        return np.concatenate(flattenedfilters + self.deforms + [self.bias])

    def size(self):
        """ Returns a DPMSize object describing the size of the DPM object,
            i.e. the information which should be unchanged by training the model.
        """
        return DPMSize(self.root.shape, map(lambda part: part.shape, self.parts))

    def __eq__(self, other):
        return (np.array_equal(self.root, other.root) 
                and np.array_equal(self.parts, other.parts)
                and np.array_equal(self.anchors, other.anchors)
                and np.array_equal(self.deforms, other.deforms)
                and self.bias == other.bias)

class DPMSize:
    """ Class describing the information of a DPM which are unchanged by
        training: number of parts, filter sizes and anchor positions.
    """
    def __init__(self, rootshape, partshapes, anchors):
        self.rootshape = rootshape
        self.partshapes = partshapes
        self.anchors = anchors

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
        return np.concatenate(map(lambda comp: comp.tovector(), self.dpms))

    def __eq__(self, other):
        return reduce(lambda b1, b2: b1 and b2,
                      map(lambda comp1, comp2: comp1 == comp2,
                          [(c1,c2) for c1 in self.dpms and c2 in other.dpms]))

    

class MixtureSize:
    """ Describes the information of a mixture model that is unchanged by training,
        namely the size and number of each component.
    """

    def __init__(dpmsizes):
        self.dpmsizes = dpmsizes

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
        filters.append(vector[offset:offset+filtersize].reshape(filtershape, order='C'))
        offset = offset + filtersize
    
    deforms = []
    # Then just take the deformation coefficients 4 by 4
    while offset < vector.size:
        deforms.append(vector[offset:offset+4])
        offset = offset + 4
    # and the bias is the last element
    bias = vector[vector.size-1]

    # build the final DPM
    return DPM(filters[0], filters[1:], dpmsize.anchors, deforms, bias)

def vectortomixture(vector, mixturesize):
    """ Converts a model vector to the corresponding mixture model.
    """
