""" Mixture DPM matching algorithm, as described by Felzenszwalb and Girshick.
"""
import cv2
import numpy as np
import skimage.feature as skifeat
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
    # pad with zeroes
    frows, fcols = featmap.shape[0:2]
    lrows, lcols = linfilter.shape[0:2]
    hrows, hcols = (lrows//2, lcols//2)
    padded = np.zeros([frows + lrows - 1, fcols + lcols - 1, featmap.shape[2]], 
                      dtype=np.float32)
    padded[hrows:hrows+frows,hcols:hcols+fcols] = featmap

    return cv2.matchTemplate(padded, 
                             linfilter.astype(np.float32), 
                             cv2.TM_CCORR)

def dpm_matching(pyramid, dpmmodel):
    """ Computes the matching of a DPM (non mixture nodel) on a
        feature pyramid.

    Arguments:
        pyramid    pyramid to match the model on.
        dpmmodel   DPM (non mixture) model to match on the pyramid.

    Returns:
        (s, p0, displacemaps)
        Where s is the score of the model on the pyramid, p0 is the 
        corresponding root position and displacemaps is an array indicating
        at 2*(p0) + anchor[i] the optimal position for part i.
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
    gdtpartresp = []
    displacemaps = []

    for i in range(0, len(dpmmodel.parts)):
        partresp = partresponses[i]
        (df,args) = gdt.gdt2D(dpmmodel.deforms[i], -partresp)
        gdtpartresp.append(-df) 
        displacemaps.append(args)
    # shift the part maps by the relative anchor position
    shiftedparts = []
    
    for i in range(0,len(dpmmodel.parts)):
        shiftedparts.append(shift(gdtpartresp[i], -dpmmodel.anchors[i]))

    # Resize the root to part resolution
    resizedrootresp = cv2.resize(rootresponse,
                                 tuple(pyramid.features[0].shape[0:2][::-1]),
                                 interpolation=cv2.INTER_NEAREST)
    # Sum it all along with bias value
    scoremap = dpmmodel.bias + resizedrootresp

    for part in shiftedparts:
        scoremap = scoremap + part

    # Compute the score as the maximum value in the score map, and the root position
    # as the position of this maximum.
    ri, rj = np.unravel_index(scoremap.argmax(), scoremap.shape)
    # Since we match the template by moving its center along the image, and
    # we expect the position of the top left corner in the rest of the code,
    # we need to convert it back
    rootpos = np.array((rj - (dpmmodel.root.shape[1]//2)*2,
                        ri - (dpmmodel.root.shape[0]//2)*2), np.int32)
    
    return (scoremap[ri, rj], rootpos, displacemaps)

def mixture_matching(pyramid, mixture):
    """ Matches a mixture model against a feature pyramid, and computes
        the latent vector corresponding to the best object hypothesis.
    
    Arguments:
        pyramid    pyramid to match the model on
        mixture    model to match on the pyramid
    Returns:
        (score, c, latvec) where:
        - score is the score of the best component
        - c is the index of the component in the mixture
        - latvec is the latent vector of the best object hypothesis for c
    """
    bestcomp = None

    # First compute individual component matchings, keep the highest
    # scoring one's root position and part respnses.
    for i in range(0, len(mixture.dpms)):
        comp = mixture.dpms[i]
        score, rootpos, displacemaps = dpm_matching(pyramid, comp)
        if bestcomp == None or bestcomp[0] < score:
            bestcomp = (score, comp, rootpos, displacemaps, i)
    
    # Compute optimal displacements for each part . Also compute the 
    # latent vector size while we're at it.
    absolutepos = []
    displacements = []
    score, comp, rootpos, displacemaps, c = bestcomp
    # initialize at root filter size + deformations + bias
    latvecsize = comp.root.size + 4 * (len(comp.parts)) + 1

    for i in range(0, len(comp.parts)):
        ancj, anci = rootpos + comp.anchors[i]
        absolutepos.append(displacemaps[i][anci, ancj])
        displacements.append(
            np.array([anci,ancj], dtype=np.int32) -
            displacemaps[i][anci, ancj]
        )
        latvecsize = latvecsize + comp.parts[i].size
    
    # Compute the latent vector for the component, required for gradient
    # computation in stochastic gradient descent.
    latvec = np.empty([latvecsize])
    offset = 0
    # add pyramid subwindows relative to each filter
    # root filter
    rrows, rcols = comp.root.shape[0:2]
    # As the filters may only be overlapping part of the image, pad the
    # feature maps with zeros.
    rpad1 = rrows//2 + 1
    cpad1 = rcols//2 + 1
    feat1pad = np.pad(pyramid.features[1], 
                      [(rpad1,rpad1), (cpad1,cpad1), (0,0)],
                      mode='constant',
                      constant_values=(0,0))
    rulj, ruli = rootpos // 2
    latvec[0:comp.root.size] = (
        feat1pad[rpad1+ruli:rpad1+ruli+rrows,
                 cpad1+rulj:cpad1+rulj+rcols]
    ).flatten('C')
    offset = offset + comp.root.size;
    # part filters
    # first pad the 0 feature map with zeros using the largest
    # part size.
    if not (comp.parts == []):
        rpad0 = np.amax(map(lambda p: p.shape[0], comp.parts))//2 + 1
        cpad0 = np.amax(map(lambda p: p.shape[1], comp.parts))//2 + 1
        
        feat0pad = np.pad(pyramid.features[0],
                          [(rpad0,rpad0), (cpad0, cpad0), (0,0)],
                          mode='constant',
                          constant_values=(0,0))

        for i in range(0, len(comp.parts)):
            part = comp.parts[i]
            prows, pcols = part.shape[0:2]
            uli, ulj = absolutepos[i]
            subwindow = feat0pad[rpad0+uli:rpad0+uli+prows,
                                 cpad0+ulj:cpad0+ulj+pcols]
            latvec[offset:offset+part.size] = subwindow.flatten('C')
            offset = offset + part.size
    
        # deformation
        for i in range(0, len(comp.parts)):
            dx = displacements[i][1]
            dy = displacements[i][0]
            latvec[offset:offset+4] = -np.array([dx, dy, dx**2, dy**2])
            offset = offset + 4

    # bias
    latvec[len(latvec) - 1] = 1

    # zero padding of the latent vector for the entire mixture
    leftpad = 0
    
    for i in range(0, c):
        leftpad = leftpad + mixture.dpms[i].size().vectorsize()

    rightpad = 0
    for i in range(c+1, len(mixture.dpms)):
        rightpad = rightpad + mixture.dpms[i].size().vectorsize()

    paddedlatvec = np.pad(latvec, [(leftpad, rightpad)], mode='constant',
                          constant_values=(0,0))

    return (score, c, paddedlatvec)
