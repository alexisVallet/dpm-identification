""" Testing for multi-class training.
"""
import unittest
import cv2
import numpy as np
import os
import json

import bintraining as btrain
import features as feat

class TestBinaryTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Loads the entire training data.
        """
        cls.character_names = [
            'asuka_langley',
            'rei_ayanami',
            'miku_hatsune',
            'monkey_d_luffy',
            'roronoa_zoro',
            'uzumaki_naruto',
            'sakura_haruno',
            'phoenix_wright',
            'maya_fey',
            'suzumiya_haruhi',
            'asahina_mikuru',
            'ginko',
            'yu_narukami',
            'naoto_shirogane',
            'shigeru_akagi',
            'kaiji',
            'lain',
            'faye_valentine',
            'radical_edward',
            'motoko_kusanagi'
        ]
        cls.imgroot = './data/images/source/'
        cls.bbroot = './data/json/boundingboxes/'
        cls.images = {}
        
        for cname in cls.character_names:
            cfiles = [f for f in os.listdir(cls.imgroot)
                      if cname in f]
            cls.images[cname] = []
            print "loading images for " + cname + "..."

            for filename in cfiles:
                bgrimg = cv2.imread(
                    os.path.join(cls.imgroot, filename)
                )
                # load the corresponding bounding boxes
                jsonfile = open(
                    os.path.join(
                        cls.bbroot, 
                        os.path.splitext(filename)[0] + '_bb.json'
                    )
                )
                boxes = json.load(jsonfile)
                jsonfile.close()
                # If there are multiple boxes in the image,
                # assume the first one is the right one.
                [[x1,y1],[x2,y2]] = boxes[0]
                # crop the image
                charbgrimg = bgrimg[y1:y2,x1:x2]
                
                cls.images[cname].append(charbgrimg)
        
    def test_binary_train(self):
        # params
        nbbins = (4,4,4)
        feature = feat.bgrhistogram(nbbins)
        featdim = np.prod(nbbins)
        mindimdiv = 7
        # set up training data
        cname = 'asuka_langley'
        positives = self.images[cname]
        negatives = []

        for othername in self.images:
            if othername != cname:
                negatives += self.images[othername]
        
        model = btrain.binary_train(
            positives,
            negatives,
            feature,
            featdim,
            mindimdiv,
            C=0.1,
            verbose=True
        )

if __name__ == "__main__":
    unittest.main()
