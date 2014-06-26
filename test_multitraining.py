""" Testing for multi-class training.
"""
import unittest
import cv2
import numpy as np
import os
import json
import multiprocessing as mp

import multitraining as mtrain
import features as feat

class TestMultiTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Loads the entire training data.
        """
        cls.character_names = [
            'asuka_langley',
            'rei_ayanami'#,
            # 'miku_hatsune',
            # 'monkey_d_luffy',
            # 'roronoa_zoro',
            # 'uzumaki_naruto',
            # 'sakura_haruno',
            # 'phoenix_wright',
            # 'maya_fey',
            # 'suzumiya_haruhi',
            # 'asahina_mikuru',
            # 'ginko',
            # 'yu_narukami',
            # 'naoto_shirogane',
            # 'shigeru_akagi',
            # 'kaiji',
            # 'lain',
            # 'faye_valentine',
            # 'radical_edward',
            # 'motoko_kusanagi'
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

    def test_multi_train(self):
        nb_parts = 6
        mindimdiv = 10
        C = 0.01
        nb_cores = 4
        models = mtrain.multi_train(
            self.images,
            nb_parts=nb_parts,
            mindimdiv=mindimdiv,
            C=C,
            verbosity=1,
            nb_cores=nb_cores
        )
        featvis = feat.bgrhistvis((4,4,4))
        # display the models
        for label in models:
            print "Displaying model for " + label
            print repr(models[label])
            mvis = models[label].toimages(featvis)
            ci = 0
            for compvis in mvis:
                pi = 0
                for partvis in compvis:
                    winname = "part " + repr(pi) + " of root " + repr(ci) 
                    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
                    cv2.imshow(winname, partvis)
                    pi += 1
                ci += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def test_train_and_save(self):
        filename = 'data/dpmid-models/testmodel'
        model1 = mtrain.train_and_save(filename, self.images)
        model2 = mtrain.load_model('data/dpmid-models/testmodel')
        self.assertEqual(model1, model2)

if __name__ == "__main__":
    unittest.main()
