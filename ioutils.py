""" Utility functions to load datasets
"""
import os
import numpy as np
import cv2
import json
import os
import os.path
from scipy.ndimage import imread

def load_data(imgfolder, bbfolder):
    # build the training data
    imagefiles = [f for f in os.listdir(imgfolder)
                  if f.endswith('.jpg')]
    traindata = {}
    
    for imgfile in imagefiles:
        stem = os.path.splitext(imgfile)[0]
        tokens = imgfile.split('_')
        # the last tokens should be 01.json which we discard.
        # just concatenate the rest again.
        label = reduce(lambda w1, w2: w1 + "_" + w2, 
                       tokens[0:len(tokens)-1])
        image = cv2.imread(os.path.join(imgfolder, imgfile))
        bboxfile = open(os.path.join(bbfolder, stem + '_bb.json'))
        bboxes = json.load(bboxfile)
        bboxfile.close()
        # we assume the character of interest is in the first
        # bounding box
        [[x1, y1], [x2, y2]] = bboxes[0]
        charimage = image[y1:y2,x1:x2]
        if not label in traindata:
            traindata[label] = [charimage]
        else:
            traindata[label].append(charimage)
    return traindata

def load_data_pixiv(folder, names=None):
    if names == None:
        names = os.walk(folder).next()[1]
    images = []
    labels = []

    for subfolder in names:
        print "Loading for " + subfolder
        imagefiles = [f for f in os.listdir(os.path.join(folder, subfolder))
                      if f.endswith('.jpg')]
        for imgfile in imagefiles:
            image = cv2.imread(os.path.join(folder, subfolder, imgfile))
            if image == None:
                print "Error loading " + imgfile
                raise Exception()
            images.append(image)
        labels += [subfolder] * len(imagefiles)
    return (images, labels)
        
