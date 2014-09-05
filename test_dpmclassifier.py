import unittest
import numpy as np
import cv2
import cPickle as pickle
import os.path

from dpm_classifier import DPMClassifier
from ioutils import load_data
from features import Combine, BGRHist, HoG

class TestDPMClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "loading data..."
        # Loads the training and testdata.
        cls.testdata = load_data(
            'data/images/5-fold/0/positives/',
            'data/json/boundingboxes/'
        )
        cls.traindata = {}

        for k in range(1,5):
            folddata = load_data(
                'data/images/5-fold/' + repr(k) + '/positives/',
                'data/json/boundingboxes/'
            )
            for label in folddata:
                if not label in cls.traindata:
                    cls.traindata[label] = folddata[label]
                else:
                    cls.traindata[label] += folddata[label]

    def test_binary_dpm_classifier(self):
        nbbins = (4,4,4)
        feature = Combine(
            BGRHist(nbbins, 0),
            HoG(9,1)
        )
        mindimdiv = 10
        C = 0.01
        nbparts = 4
        deform_factor = 1.
        classifier = DPMClassifier(
            C,
            feature,
            mindimdiv,
            nbparts,
            deform_factor,
            nb_coord_iter=4,
            nb_gd_iter=25,
            learning_rate=0.001,
            inc_rate=1.2,
            dec_rate=0.5,
            use_pca=0.9,
            verbose=True
        )

        trainsamples = []
        trainlabels = []

        for k in self.traindata:
            for s in self.traindata[k]:
                trainsamples.append(s)
                trainlabels.append(k)

        print "Training..."
        cachename = 'data/dpmid-cache/test'
        if not os.path.isfile(cachename):
            classifier.train_validation_named(trainsamples, trainlabels)
            cachefile = open(cachename, 'w')
            pickle.dump(classifier, cachefile)
            cachefile.close()
        else:
            cachefile = open(cachename)
            classifier = pickle.load(cachefile)
            cachefile.close()

        print "Deformation coeffs:"
        for dpm in classifier.dpms:
            print dpm.deforms

        print "Prediction..."

        testsamples = []
        expected = []

        for k in self.testdata:
            for s in self.testdata[k]:
                testsamples.append(s)
                expected.append(k)
        probas = classifier.predict_proba(testsamples)
        nb_classes = probas.shape[1]
        
        for top in range(1, 21):
            nb_correct = 0

            for i in range(len(testsamples)):
                top_idx = np.argsort(probas[i,:])[nb_classes-top:].tolist()
                top_names = map(lambda i: classifier.int_to_label[i], top_idx)

                if expected[i] in top_names:
                    nb_correct += 1

            print "top " + repr(top) + " recognition rate: " + repr(float(nb_correct)/len(testsamples))

if __name__ == "__main__":
    unittest.main()
