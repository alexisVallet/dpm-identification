""" Mixture DPM matching algorithm, as described by Felzenszwalb and Girshick.
"""
import cv2
import numpy as np
import skimage.feature as skifeat
import gdt
import dpm
import featpyramid as fpyr
import features as feat

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

def part_resp_pad(comp):
    """ Computes the amount of padding necessary for parts matching with
        a specific DPM.

    Arguments:
        featmap    feature map to pad.
        comp       DPM to match on the feature map.

    Returns:
        (pl, pr, pt, pb) where pl, pr, pt and pb are the amount of padding 
        on the left, right, top and bottom respectively.
    """
    pl = 0
    pr = 0
    pt = 0
    pb = 0

    for i in range(len(comp.parts)):
        hsize = np.array(comp.parts[i].shape[0:2], np.int32)//2
        anchor = comp.anchors[i]
        pl = min(pl, anchor[1] - hsize[1])
        pr = max(pr, anchor[1] + hsize[1])
        pt = min(pt, anchor[0] - hsize[0])
        pb = max(pb, anchor[0] + hsize[0])

    pl = -pl
    pt = -pt

    # More padding doesn't really hurt (except in memory usage,
    # which we have plenty of anyway) so we add 1 to avoid any
    # off-by-one error somewhere.
    pl += 1
    pr += 1
    pt += 1
    pb += 1

    return (pl, pr, pt, pb)

def dpm_matching(pyramid, dpmmodel, comp):
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
    
    # Modification from Felzenszwalb and Girshick: we do not actually
    # slide the root filter, but just warp the source image to the root
    # size. This just gives one root response value. Since we roughly
    # already know where the root is thanks to the bounding box, setting
    # it as a latent value seems to introduce more noise than signal.
    rootresponse = np.vdot(pyramid.rootfeatures[comp], dpmmodel.root)
    # Apply 0-padding to the parts feature map, so everything can be
    # correctly matched. Keep track of the amount of padding.
    pl, pr, pt, pb = part_resp_pad(dpmmodel)
    ftrows, ftcols = pyramid.features[0].shape[0:2]
    paddedfmap = np.pad(
        pyramid.features[0],
        [(pt,pb),(pl,pr),(0,0)],
        mode='constant',
        constant_values=[(0,0),(0,0),(0,0)]
    )
    partresponses = map(lambda part:
                        filter_response(paddedfmap, part),
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
        (df,args) = gdt.gdt2D(dpmmodel.deforms[i], -partresponses[i])
        # we only care about the GDT'd response for the center part, so
        # we remove the padding
        gdtpartresp.append(-df[pt:pt+ftrows,pl:pl+ftcols])
        # but we care about the args indexes with padding
        displacemaps.append(args)
    # shift the part maps by the relative anchor position
    shiftedparts = []
    
    for i in range(0,len(dpmmodel.parts)):
        shiftedparts.append(shift(gdtpartresp[i], -dpmmodel.anchors[i]))

    # Sum it all along with bias value
    scoremap = dpmmodel.bias + rootresponse

    for part in shiftedparts:
        scoremap = scoremap + part

    # Compute the score as the maximum value in the score map. No root
    # position necessary, just part displacements.
    return (scoremap.max(), displacemaps, pl, pt, paddedfmap)

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
        score, displacemaps, pl, pt, paddedfmap = (
            dpm_matching(pyramid, comp, i)
        )
        if bestcomp == None or bestcomp[0] < score:
            bestcomp = (score, comp, displacemaps, pl, pt, paddedfmap, i)
    
    # Compute optimal displacements for each part . Also compute the 
    # latent vector size while we're at it.
    absolutepos = []
    displacements = []
    score, comp, displacemaps, pl, pt, paddedfmap, c = bestcomp
    # initialize at root filter size + deformations + bias
    latvecsize = comp.root.size + 4 * (len(comp.parts)) + 1

    for i in range(0, len(comp.parts)):
        # computing displacements: the displacement maps are
        # padded by pl on the left and pt on the top, so careful
        # with that one.
        # anchors are specified as upper-left point, but responses
        # correspond to the center of the filter. So we shift it
        # by half the filter size.
        ancj, anci = (
            np.array([pl, pt], np.int32) + # padding
            comp.anchors[i] + # anchor position
            np.array(comp.parts[i].shape[0:2][::-1], np.int32)//2 # half size
        )
        # the absolute position is padded coordinates - needed for
        # reconstructing the latent vector later on. It's also still
        # coordinate for the center of the filter.
        absolutepos.append(displacemaps[i][anci, ancj])
        # no need to take padding into account for the displacement
        # itself, as we only care about the difference.
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
    latvec[0:comp.root.size] = pyramid.rootfeatures[c].flatten('C')
    offset = offset + comp.root.size;

    if not (comp.parts == []):
        # part filters, being careful to take subwindow out of the
        # well-padded feature map.
        # Since deformations may put us out of bounds by a halfsize,
        # we add that much more padding.
        maxhsr = max(map(lambda p: p.shape[0], comp.parts))//2
        maxhsc = max(map(lambda p: p.shape[1], comp.parts))//2
        repaddedfmap = np.pad(
            paddedfmap,
            [(maxhsr,maxhsr),(maxhsc,maxhsc),(0,0)],
            mode='constant',
            constant_values=[(0,0),(0,0),(0,0)]
        )
        for i in range(0, len(comp.parts)):
            part = comp.parts[i]
            prows, pcols = part.shape[0:2]
            # absolutepos contains the absolute position of
            # the center of the part, so we gotta be careful
            # again
            uli, ulj = absolutepos[i]
            subwindow = repaddedfmap[uli:uli+prows,
                                     ulj:ulj+pcols]
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
