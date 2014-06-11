import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import cv2
import sys

lib = ctypes.cdll.LoadLibrary("./c/gdt.so")
gdt1D = lib.gdt1D
gdt1D.restype = None
gdt1D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ctypes.c_int,
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]

def distancetransform(d, f):
    """ 2 dimensional distance transform of function f for quadratic distance measure.
    
    Args:
        d    4 elements numpy vector specifying the distance function 
             dist(dx, dy) = d . (dx, dy, dx^2, dy^2)
        f    function to compute the distance transform of. 2d numpy array where f[i,j]
             is the value of the function at row i and column j.
    Returns:
       (df, arg)
       where df is an array of the same shape as f specifying its distance transform,
       and arg is a list of indexes corresponding to the argmax version of the problem.
    """
    # First run 1D transform along the x axis - i.e. on f flattened row-major
    dx = np.array([d[0], d[2]]).astype(np.float32)
    n = f.size
    df1 = np.empty(n, dtype=np.float32)
    arg = np.empty(n, dtype=np.int32)
    gdt1D(dx, f.flatten('C').astype(np.float32), n, df1, arg)
    # Then along the y axis
    dy = np.array([d[1], d[3]]).astype(np.float32)
    df2 = np.empty(n, dtype=np.float32)
    gdt1D(dy, np.reshape(df1, (f.shape[0], f.shape[1]), order='C').flatten('F'), n, 
          df2, arg)
    # unravel the arg indices.
    unraveledarg = np.array(np.unravel_index(arg, f.shape, order='F')).reshape(f.shape, order='F')

    return (np.reshape(df2, f.shape, order='F'), unraveledarg)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please input an image")
    img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    (dfimg, args) = distancetransform(np.array([0,0,1,1]), img)
    cv2.imshow("distance transform", dfimg / 255)
    hopefully = np.empty(img.shape)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            hopefully[i,j] = dfimg[args[i,j]]
    cv2.imshow("hopefully also gdt", hopefully)
    cv2.waitKey(0)
