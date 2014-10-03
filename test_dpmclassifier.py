# -*- coding: utf-8 -*-
import unittest
import numpy as np
import cv2
import cPickle as pickle
import os.path

from dpm_classifier import DPMClassifier
from ioutils import load_data, load_data_pixiv
from features import Combine, BGRHist, HoG
from cross_validation import k_fold_split

class TestDPMClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = [
            "初音ミク",
            "鏡音リン",
            "本田菊",
            "チルノ",
            "鏡音レン",
            "アーサー・カークランド",
            "レミリア",
            "暁美ほむら",
            "アリス",
            "霧雨魔理沙",
            "ルーミア",
            "黒子テツヤ",
            "美樹さやか",
            "巡音ルカ",
            "ギルベルト・バイルシュミット",
            "フランドール・スカーレット",
            "坂田銀時",
            "古明地こいし",
            "東風谷早苗",
            "アルフレッド・F・ジョーンズ"
        ]
        print "loading data..."
        images, labels = load_data_pixiv('data/pixiv-images-1000', names)
        print "finished loading data."
        fold_samples, fold_labels = k_fold_split(images, labels, 3)
        cls.traindata = reduce(lambda l1,l2: l1 + l2, fold_samples[1:])
        cls.trainlabels = reduce(lambda l1,l2: l1 + l2, fold_labels[1:])
        cls.testdata = fold_samples[0]
        cls.testlabels = fold_labels[0]
        print repr(len(cls.traindata)) + " training samples"
        print repr(len(cls.testdata)) + " test samples"
        print repr(len(images)) + " total"

    def test_binary_dpm_classifier(self):
        nbbins = (4,4,4)
        feature = Combine(
            BGRHist(nbbins, 0),
            HoG(9,0)
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
            nb_gd_iter=50,
            learning_rate=0.001,
            inc_rate=1.2,
            dec_rate=0.5,
            cst_deform=None,
            use_pca=0.9,
            verbose=True
        )

        print "Training..."
        classifier.train_named(self.traindata, self.trainlabels)
        print "Top-1 to top-20 accuracy:"
        print classifier.top_accuracy_named(self.testdata, self.testlabels)

if __name__ == "__main__":
    unittest.main()
