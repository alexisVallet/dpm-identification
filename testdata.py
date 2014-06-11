import dpm 
import numpy as np
import numpy.random as npr
import featpyramid as pyr

def randomdpm(pyramid, nbparts):
    """ Generate a random DPM from a feature pyramid, given a specific
        number of parts.
    """
    # Take a random subwindow of the feature map as root filter
    rows, cols = pyramid.features[1].shape[0:2]
    rootrows, rootcols = (npr.randint(1,rows),npr.randint(1,cols))
    ruli, rulj = (npr.randint(0, rows-rootrows), npr.randint(0,cols-rootcols))
    root = pyramid.features[1][ruli:ruli+rootrows,rulj:rulj+rootcols]
    
    # Randomly place subwindows on the root filter to get parts
    parts = []
    anchors = []
    deforms = []

    for i in range(0,nbparts):
        partrows = npr.randint(1,max(2,rootrows))
        partcols = npr.randint(1,max(2,rootcols))
        puli, pulj = (npr.randint(0,max(1, rootrows - partrows)),
                      npr.randint(0,max(1, rootcols - partcols)))
        # convert all these relative coords to the right scale and displacement
        puli = (ruli + puli) * 2
        pulj = (rulj + pulj) * 2
        partrows = partrows * 2
        partcols = partcols * 2
        parts.append(pyramid.features[0][puli:puli+partrows,pulj:pulj+partcols])
        anchors.append(np.array([pulj,puli]))
        # random deformations
        deforms.append(np.absolute(npr.rand(4)))
    return dpm.DPM(root, parts, anchors, deforms, npr.random())

def randommixture(pyramid):
    """ Generate a random mixture model from a feature pyramid.
    """
    return dpm.Mixture(map(lambda i: randomdpm(pyramid, npr.randint(0,5)), range(0,npr.randint(0,5))))
