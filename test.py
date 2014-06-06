import skimage.util as skiutil
import numpy as np
import cv2
import sys
import math

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    h, w = img.shape[0:2]
    nbcells = 20
    cellsize = math.sqrt(w*h/nbcells)
    wa = w // cellsize
    wnbsteps = wa if w % cellsize == 0 else wa + 1
    ha = h // cellsize
    hnbsteps = ha if h % cellsize == 0 else ha + 1
    wstep = w / wnbsteps
    hstep = h / hnbsteps
    
    for i in range(0, int(hnbsteps)):
        for j in range(0, int(wnbsteps)):
            uli = round(i * wstep)
            ulj = round(j * wstep)
            window = img[uli:uli+cellsize,ulj:ulj+cellsize]
            cv2.imshow("window", window)
            cv2.waitKey(0)
