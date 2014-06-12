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

gdt2D = lib.gdt2D
gdt2D.restype = None
gdt2D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ctypes.c_int,
                  ctypes.c_int,
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
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
    arg1 = np.empty(n, dtype=np.int32)
    gdt1D(dx, f.flatten('C').astype(np.float32), n, df1, arg1)
    # Then along the y axis
    dy = np.array([d[1], d[3]]).astype(np.float32)
    df2 = np.empty(n, dtype=np.float32)
    arg2 = np.empty(n, dtype=np.int32)
    gdt1D(dy, np.reshape(df1, (f.shape[0], f.shape[1]), order='C').flatten('F'), n, 
          df2, arg2)
    # unravel the indices for both arg1 and arg2
    rows, cols = f.shape
    unraveledarg1 = np.array(np.unravel_index(arg1, f.shape, order='C')).T.reshape([rows, cols, 2], order='C')
    unraveledarg2 = np.array(np.unravel_index(arg2, f.shape, order='F')).T.reshape([rows, cols, 2], order='F')

    # create the final args array by first looking up the index in df1 from arg2,
    # then looking up the index in f from arg1
    args = np.empty([rows,cols,2], dtype=np.int32)
    
    for i in range(0,rows):
        for j in range(0,cols):
            i1, j1 = unraveledarg2[i,j]
            i2, j2 = unraveledarg1[i1,j1]
            args[i,j] = np.array([i2,j2], dtype=np.int32)

    return (np.reshape(df2, f.shape, order='F'), args)

def cgdt2D(d, f):
    df = np.empty(f.shape, dtype=np.float32, order='C')
    argi = np.empty(f.shape, dtype=np.int32, order='C')
    argj = np.empty(f.shape, dtype=np.int32, order='C')
    gdt2D(d.astype(np.float32), f.astype(np.float32), f.shape[0], 
          f.shape[1], df, argi, argj)
    
    return (df, np.concatenate((argi, argj), axis=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please input an image")
    img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    (dfimg, args) = cgdt2D(np.array([0,0,1,1]), img)
    cv2.imshow("distance transform", dfimg / 255)
    hopefully = np.empty(img.shape, dtype=np.float32, order='C')
    print repr(img.shape)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            i1, j1 = args[i,j]
            imi1 = i - i1
            jmj1 = j - j1
            dist = imi1*imi1 + jmj1*jmj1 
            if dist > 255:
                print "dist: " + repr(dist)
                print "(i,j): " + repr((i,j)) + ", (i1, j1): " + repr((i1,j1))
            hopefully[i,j] = img[i1, j1] + dist
    cv2.imshow("hopefully also gdt", hopefully/255)
    cv2.waitKey(0)
