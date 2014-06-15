""" Mixture DPM matching algorithm, as described by Felzenszwalb and Girshick.
"""
import cv2
import numpy as np
import scipy.ndimage.filters as filters
import gdt
import dpm
import featpyramid as fpyr

def shift(array, vector):
    """ Shift array coefficients, filling with zeros.
    
    Arguments:
        array    array to shift.
        vector   vector to shift array coefficients by.
    
    Returns:
       An array of the same size as the inputs, such that:
          shift(array)[i,j] = array[i+vector[1],j+vector[0]]
       If (i,j)+vector is not inside the source array, it is filled with 
       zeros. Vector can have negative components.
    """
    rows, cols = array.shape
    assert abs(vector[1]) < rows and abs(vector[0]) < cols
    # shift the data
    shifted = np.roll(np.roll(array, vector[1], 0), vector[0], 1)
    # pad with zeros
    # positive displacement
    if vector[1] > 0:
        shifted[0:vector[1],:] = 0
    # negative displacement
    elif vector[1] < 0:
        shifted[rows + vector[1]:rows,:] = 0
    # same for the other axis
    if vector[0] > 0:
        shifted[:,0:vector[0]] = 0
    elif vector[0] < 0:
        shifted[:,cols + vector[0]:cols] = 0
    return shifted

def filter_response(featmap, linfilter):
    """ Compute the response of a single filter on a feature pyramid.
    
    Arguments:
        featmap    feature map to compute the response of the filter on.
        linfilter  linear filter to match against the feature map.

    Return:
        A 2D response map of the same size as featmap.
    """
    return np.sum(filters.correlate(featmap, linfilter), axis=2)

def dpm_matching(pyramid, dpmmodel):
    """ Computes the matching of a DPM (non mixture nodel) on a
        feature pyramid.

    Arguments:
        pyramid    pyramid to match the model on.
        dpmmodel   DPM (non mixture) model to match on the pyramid.

    Returns:
        (s, p0, partresps)
        Where s is the score of the model on the pyramid, p0 is the 
        corresponding root position and partresps is a list of part 
        filter responses - necessary to compute part displacements.
    """
    # Compute the response of all the filters of the dpmmodel one by one
    # by cross correlation. The root filter is correlated to the low
    # resolution layer, and the parts to the high resolution one.
    rootresponse = filter_response(pyramid.features[1], dpmmodel.root)
    partresponses = map(lambda part: 
                        filter_response(pyramid.features[0], part),
                        dpmmodel.parts)
    # Apply distance transform to part responses.
    # The reason we do -gdt(d, -r) is because the gdt actually computes
    # an (arg)max while we want an (arg)min. A tiny bit of maths shows
    # that inverting the signs this way expresses the score function as
    # a valid generalized distance transform.
    gdtpartresp = map(lambda partresp:
                      -gdt.gdt2d(dpmmodel.deforms, -partresp)[0],
                      partresponses)
    # Resize and shift the part maps by the relative anchor position
    shiftedparts = []
    
    for i in range(0,len(dpmmodel.parts)):
        resized = cv2.resize(gdtpartresp, tuple(rootresponse.shape),
                             interpolation=cv2.INTER_NEAREST)
        shiftedparts.append(shift(resized,
                                  np.round_(-dpmmodel.anchors[i]/2).astype(np.int32)))
    # Sum it all along with bias value
    scoremap = dpmmodel.bias + rootresponse
    for part in shiftedparts:
        scoremap = scoremap + part

    # Compute the score as the maximum value in the score map, and the root position
    # as the position of this maximum.
    rootpos = scoremap.argmax()
    
    return (scoremap[rootpos], rootpos, partresponses)

def mixture_matching(pyramid, mixture):
    """ Matches a mixture model against a feature pyramid.
    
    Arguments:
        
    """
    bestcomp = None

    # First compute individual component matchings, keep the highest
    # scoring one's root position and part respnses.
    for i in range(0, len(mixture.dpms)):
        comp = mixture.dpms[i]
        score, rootpos, partresponses = dpm_matching(pyramid, comp)
        if bestcomp == None or bestcomp[0] < score:
            bestcomp = (score, comp, rootpos, partresponses, i)
    
    # Compute optimal displacements for each part using GDT on part
    # responses. Also compute the latent vector size while we're at it.
    absolutepos = []
    displacements = []
    score, comp, rootpos, partesponses, c = bestcomp
    # initialize at root filter size + deformations + bias
    latvecsize = comp.root.size + 4 * (len(comp.parts) + 1) + 1

    for i in range(0, len(comp.parts))
        (partgdt, args) = gdt2D(comp.deforms[i], -partresponses[i])
        ancj, anci = np.round(2*rootpos + comp.anchors[i]).astype(np.int32)
        absolutepos.append(args[anci, ancj])
        displacements.append(args[anci, ancj] - 
                             np.array([anci,ancj], dtype=np.int32))
        latvecsize = latvecsize + comp.parts[i].size
    
    # Compute the latent vector for the component, required for gradient
    # computation in stochastic gradient descent.
    latvec = np.empty([latvecsize])
    offset = 0
    # add pyramid subwindows relative to each filter
    # root filter
    rrows, rcols = comp.root.shape[0:2]
    latvec[0:comp.root.size] = (
        pyramid.features[1][rootpos[0]:rootpos[0]+rrows,
                            rootpos[1]:rootpos[1]+rcols]
    ).flatten('C')
    offset = offset + comp.root.size;
    # part filters
    for i in range(0, len(comp.parts)):
        part = comp.parts[i]
        prows, pcols = part.shape[0:2]
        uli, ulj = absolutepos[i]
        latvec[offset:offset+part.size] = (
            pyramid.features[0][uli:uli+prows,ulj:ulj+pcols]
        ).flatten('C')
        offset = offset + part.size
    
    # deformation 
    for i in range(0, len(comp.parts)):
        latvec[offset:offset+4] = comp.deforms[i]
        offset = offset + 4

    # bias
    latvec[len(latvec) - 1] = comp.bias

    return (score, c, latvec)
