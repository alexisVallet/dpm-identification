import ctypes
from numpy.ctypeslib import ndpointer, as_ctypes
import numpy as np
import cv2
import sys

lib = ctypes.cdll.LoadLibrary("./c/gdt.so")
cgdt2D = lib.gdt2D
cgdt2D.restype = None
cgdt2D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ndpointer(ctypes.c_float, flags=("C_CONTIGUOUS",'W')),
                   ndpointer(ctypes.c_int, flags=("C_CONTIGUOUS",'W')),
                   ndpointer(ctypes.c_int, flags=("C_CONTIGUOUS",'W'))]

def gdt2D(d, f, rg=None):
    """ Computes the generalized distance transform of a function on a 2D
        grid, with quadratic distance measure.
    
    Arguments:
        d 4-elements vector containing quadratic coefficients (ax,ay,bx,by)
          defining the following distance measure 
             d(dx, dy) = ax*dx + bx*dx^2 + ay*dy + by*dy^2
        f numpy 2D array defining function values at each point of the
          grid.

    Returns:
        (df, args) where:
        - df(p) = min_q(f(q) + d(p - q))
        - args(p) = argmin_q(f(q) + d(p - q))
    """
    rows, cols = f.shape
    if rg==None:
        rg = max(rows,cols)
    df = np.empty(f.shape, dtype=np.float32)
    argi = np.empty([rows,cols,1], dtype=np.int32)
    argj = np.empty([rows,cols,1], dtype=np.int32)
    cgdt2D(d.astype(np.float32), f.astype(np.float32), rows, cols,
           df, argi, argj, rg)

    return (df, np.concatenate((argi,argj), axis=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please input an image")
    img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    (dfimg, args) = gdt2D(np.array([0,0,1,1]), img)
    cv2.imshow("distance transform", dfimg / 255)
    cv2.waitKey(0)
