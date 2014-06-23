import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import cv2
import dsf
import features as feat

cfreegraph = dsf.felzlib.free_graph
cfreegraph.restype = None
cfreegraph.argtypes = [ctypes.c_void_p]

cgridgraph = dsf.felzlib.c_gridGraph
cgridgraph.restype = ctypes.c_void_p
cgridgraph.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

cfelzseg = dsf.felzlib.c_felzenszwalbSegment
cfelzseg.restype = ctypes.c_void_p
cfelzseg.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

def felzenszwalb_segment(featmap, k=None, mincmpsize=0.01, maxcmpsize = 0.8,
                         scaletype='CARDINALITY'):
    """ Run a modified version of Felzenszwalb's algorithm designed for
        feature maps. Internally, uses the correlation distance as the
        edge weights.

    Arguments:
        featmap    feature map to segment into parts.
    
    Returns:
        A set of rectangular parts of the feature map.
    """
    rows, cols, fdim = featmap.shape
    # default k
    if k == None:
        k = min(rows, cols) / 2
    scaleval = 1 if scaletype == 'VOLUME' else 0

    # create the appropriate graph
    graph = cgridgraph(featmap.astype(np.float32), rows, cols, fdim,
                       scaleval)
    # run the segmentation algorithm
    forest = cfelzseg(k, graph, int(2*mincmpsize*float(rows*cols)), 
                      rows, cols, scaleval)
    dsforest = dsf.DisjointSetForest(forest)
    segmentation = dsf.Segmentation(dsforest, featmap)
    parts = dsf.segment_boxes(segmentation, mincmpsize, maxcmpsize)
    # free the data
    cfreegraph(graph)
    dsf.cfreedsf(forest)

    return parts

if __name__ == "__main__":
    img = cv2.imread('data/images/source/asahina_mikuru_0.jpg')
    nbbins = (4,4,4)
    featmap = feat.compute_featmap(img, 20, 20, 
                                   feat.bgrhistogram(nbbins),
                                   np.prod(nbbins))
    parts = felzenszwalb_segment(featmap)
    print "finished computing parts"
    vis = lambda fmap: feat.visualize_featmap(fmap, feat.bgrhistvis(nbbins))
    cv2.namedWindow("root", cv2.WINDOW_NORMAL)
    cv2.imshow("root", vis(featmap))
    i = 0
    for partandanc in parts:
        (anc, part)
        print part.shape
        cv2.namedWindow("part " + repr(i), cv2.WINDOW_NORMAL)
        cv2.imshow("part " + repr(i), vis(part))
        i = i + 1
    cv2.waitKey(0)
