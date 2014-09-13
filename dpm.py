""" Deformable parts model implementation.
"""
import numpy as np
import cv2

import features as feat

class DPM:
    def __init__(self, parts, anchors, deforms):
        """ Initializes a deformable parts model with no root and a
            set of part filters with relative anchor positions.
        
        Argument:
            parts   list of k n_i by m_i by featdim numpy arrays for the 
                    parts filters, n_i and m_i possibly varying for each.
            anchors list of k anchor positions for each part, as 2d (i,j) 
                    numpy arrays corresponding to placement relative to
                    the root filter in the lower layer of the pyramid.
            deforms list of k 4d vectors containing deformation
                    coefficients.
        """
        assert len(parts) == len(anchors)

        self.parts = parts
        self.anchors = anchors
        self.deforms = deforms

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
        flattenedfilters = map(lambda f: f.flatten('C'), self.parts)
        toconcat = flattenedfilters + self.deforms

        return np.concatenate(toconcat)

    def tovector_nodeform(self):
        """ Computes a vector representation suitable for classification,
            discarding deformation coefficients.
        """
        return np.concatenate(map(lambda f: f.flatten('C'), self.parts))
    
    def size(self):
        """ Returns a DPMSize object describing the size of the DPM object,
            i.e. the information which should be unchanged by training the
            model.
        """
        return DPMSize(map(lambda part: part.shape, self.parts),
                       self.anchors)

    def __eq__(self, other):
        allpairsclose = (lambda l1, l2: 
                         reduce(lambda b, p: b and np.allclose(p[0], p[1]), 
                                zip(l1, l2), True))

        return (np.allclose(self.root, other.root) 
                and allpairsclose(self.parts, other.parts)
                and allpairsclose(self.anchors, other.anchors)
                and allpairsclose(self.deforms, other.deforms))

    def toimages(self, featvis):
        """ Returns images for the root and for each part.
        """
        return map(
            lambda fmap: feat.visualize_featmap(
                fmap, 
                featvis,
                blocksize=(1,1)), 
            self.parts)

    def partsimage(self, featvis):
        partimages = self.toimages(featvis)
        # Place the parts in a black image. First need to determine
        # the size needed.
        mini = np.inf
        maxi = -np.inf
        minj = np.inf
        maxj = -np.inf

        for i in range(len(self.parts)):
            anci, ancj = self.anchors[i]
            prows, pcols = self.parts[i].shape[0:2]
            mini = min(mini, anci)
            maxi = max(maxi, anci + prows)
            minj = min(minj, ancj)
            maxj = max(maxj, ancj + pcols)

        image = np.zeros([maxi - mini + 1, maxj - minj + 1, 3], np.float32)
        # Then place the parts one by one.
        for i in range(len(self.parts)):
            anci, ancj = self.anchors[i] - np.array([mini, minj])
            imrows, imcols = partimages[i].shape[0:2]
            image[anci:anci+imrows, ancj:ancj+imcols] = partimages[i]

        # Resize and draw rectangles so one can see parts better.
        resized = cv2.resize(
            image, None, fx=32, fy=32, 
            interpolation=cv2.INTER_NEAREST
        )
        for i in range(len(self.parts)):
            anci, ancj = (self.anchors[i] - np.array([mini, minj])) * 32
            rrows, rcols = np.array(partimages[i].shape[0:2]) * 32
            cv2.rectangle(
                resized, 
                (ancj, anci), 
                (ancj + rcols, anci + rrows),
                (0,1,0),
                4
            )

        return resized

    def deforms_images(self):
        def _deforms_image(i):
            # Plot the deformation function across a patch
            # corresponding to the size of the part, with
            # origin at the center.
            prows, pcols = self.parts[i].shape[0:2]
            plot = np.empty([prows*2, pcols*2], np.float32)
            cdi, cdj, cdi2, cdj2 = self.deforms[i]
            
            for i in range(prows*2):
                for j in range(pcols*2):
                    di = i - prows
                    dj = j - pcols
                    plot[i,j] = cdi*di + cdj*dj + cdi*di**2 + cdj*dj**2

            # Normalize to [0;1] range.
            plot = (plot - plot.min()) / (plot.max() - plot.min())
            # 1 minus plot so black is high cost and white is low cost.
            plot = 1 - plot
            
            return plot

        return map(_deforms_image, range(len(self.parts)))

    def __repr__(self):
        return repr(self.size())                    

class DPMSize:
    """ Class describing the information of a DPM which are unchanged by
        training: number of parts, filter sizes and anchor positions.
    """
    def __init__(self, partshapes, anchors):
        self.partshapes = partshapes
        self.anchors = anchors

    def vectorsize(self):
        return int(np.sum(map(lambda ps: np.prod(ps),self.partshapes)) +
                   len(self.partshapes) * 4)

    def vectorsize_nodeform(self):
        return int(np.sum(map(lambda ps: np.prod(ps),self.partshapes)))

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
        """ Returns the size of the model, i.e. the information unchanged 
            by training the model.

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

    def toimages(self, featvis):
        return map(lambda m: m.toimages(featvis), self.dpms)

    def __repr__(self):
        return repr(self.size())

class MixtureSize:
    """ Describes the information of a mixture model that is unchanged by 
        training, namely the size and number of each component.
    """
    def __init__(self, dpmsizes):
        self.dpmsizes = dpmsizes

    def vectorsize(self):
        return sum(map(lambda psize: psize.vectorsize(), self.dpmsizes))

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

    for filtershape in dpmsize.partshapes:
        filtersize = np.prod(filtershape)
        subvec = vector[offset:offset+filtersize]
        filters.append(subvec.reshape(filtershape, order='C'))
        offset = offset + filtersize
    
    deforms = []
    # Then just take the deformation coefficients 4 by 4
    while offset < vector.size:
        deforms.append(vector[offset:offset+4])
        offset = offset + 4

    # build the final DPM
    return DPM(filters, dpmsize.anchors, deforms)

def vectortomixture(vector, mixturesize):
    """ Converts a model vector to the corresponding mixture model.
    """
    offset = 0
    dpms = []

    for dpmsize in mixturesize.dpmsizes:
        # The size of the subvector is the sum of filter sizes, deformation
        # sizes and the bias
        subvecsize = dpmsize.vectorsize()
                      
        dpms.append(vectortodpm(vector[offset:offset+subvecsize],
                                dpmsize))
        offset = offset + subvecsize
    return Mixture(dpms)
