""" Wrapper around the disjoint set forest implementation used for
    Felzenszwalb's segmentation algorithm.
"""
import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

# low level wrapper
felzlib = ctypes.cdll.LoadLibrary(
    "./felzenszwalb-segmentation/libfelzseg.so"
)

cnewdsf = felzlib.new_DisjointSetForest
cnewdsf.restype = ctypes.c_void_p
cnewdsf.argtypes = [ctypes.c_int]

cfreedsf = felzlib.free_dsf
cfreedsf.restype = None
cfreedsf.argtypes = [ctypes.c_void_p]

cfind = felzlib.find
cfind.restype = ctypes.c_int
cfind.argtypes = [ctypes.c_void_p, ctypes.c_int]

cunion = felzlib.setUnion
cunion.restype = ctypes.c_int
cunion.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int
]

cnbcomps = felzlib.getNumberOfComponents
cnbcomps.restype = ctypes.c_int
cnbcomps.argtypes = [ctypes.c_void_p]

ccompsize = felzlib.getComponentSize
ccompsize.restype = ctypes.c_int
ccompsize.argtypes = [ctypes.c_void_p, ctypes.c_int]

class DisjointSetForest:
    """ Low level wrapper around the disjoint set forest C++ class.
    """

    def __init__(self, forest):
        """ Initializes a disjoint set forest containing a certain
            number of elements. Each of them is in its own singleton
            component initially.

        Arguments:
            nbelems the total number of elements.
        """
        self.forest = forest

    def find(self, elem):
        """ Return the index of the root element of the component containing
            a given element.

        Arguments:
            elem    element to find the root of.
        """
        return cfind(self.forest, elem)
    
    def union(self, e1, e2):
        """ Fuses sets containing elements e1 and e2. Returns the new root
            index.
        
        Arguments:
            e1, e2    elements of the sets to fuse.
        """
        return cunion(self.forest, e1, e2)

    def nb_comps(self):
        """ Returns the total number of components in the forest.
        """
        return cnbcomps(self.forest)

    def comp_size(self, elem):
        """ Return the size (i.e. number of elements) of the component
            of a 
        """
        return ccompsize(self.forest, elem)

class Segmentation:
    """ Segmentation of a given feature map or image.
    """
    def __init__(self, dsf, fmap):
        self.dsf = dsf
        self.fmap = fmap

    def find(self, i, j):
        flatidx = self.dsf.find(np.ravel_multi_index(
            [i,j], self.fmap.shape[0:2], order='C'
        ))
        
        return np.unravel_index(
            flatidx, 
            self.fmap.shape[0:2],
            order='C'
        )

def segment_boxes(segmentation, minsize, maxsize):
    """ Return bounding box subwindows for each se ment in a
        a segmentation.
    """
    # compute boxes bounds
    boxes = {}
    rows, cols = segmentation.fmap.shape[0:2]

    for i in range(rows):
        for j in range(cols):
            root = segmentation.find(i,j)
            if not root in boxes:
                boxes[root] = [cols, rows, 0, 0]
            x1, y1, x2, y2 = boxes[root]
            boxes[root] = [min(j,x1),min(i,y1),max(j,x2),max(i,y2)]
    
    # compute appropriate subwindows
    subwins = []

    for root in boxes:
        x1, y1, x2, y2 = boxes[root]
        area = (x1 - x2) * (y1 - y2)
        ratio = float(area) / float(rows * cols)
        if minsize <= ratio <= maxsize:
            subwins.append(
                (np.array((x1,y1), np.int32), 
                 segmentation.fmap[y1:y2+1,x1:x2+1])
            )

    return subwins
