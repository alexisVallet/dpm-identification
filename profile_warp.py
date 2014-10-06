# -*- coding: utf-8 -*-
import numpy as np

from warpclassifier import WarpClassifier
from ioutils import load_data, load_data_pixiv
from features import Combine, BGRHist, HoG
from cross_validation import k_fold_split

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
    "霧雨魔理沙"# ,
            # "ルーミア",
            # "黒子テツヤ",
            # "美樹さやか",
            # "巡音ルカ",
            # "ギルベルト・バイルシュミット",
            # "フランドール・スカーレット",
            # "坂田銀時",
            # "古明地こいし",
            # "東風谷早苗",
            # "アルフレッド・F・ジョーンズ"
]
print "loading data..."
images, labels = load_data_pixiv('data/pixiv-images-1000', names)
print "finished loading data."
fold_samples, fold_labels = k_fold_split(images, labels, 3)
traindata = reduce(lambda l1,l2: l1 + l2, fold_samples[1:])
trainlabels = reduce(lambda l1,l2: l1 + l2, fold_labels[1:])
testdata = fold_samples[0]
testlabels = fold_labels[0]
print repr(len(traindata)) + " training samples"
print repr(len(testdata)) + " test samples"
print repr(len(images)) + " total"
          
# Run training.
nbbins = (4,4,4)
feature = Combine(
    HoG(5, 1),
    BGRHist(nbbins, 0)
)
mindimdiv = 8
C = 0.01
classifier = WarpClassifier(
    feature,
    mindimdiv,
    C,
    learning_rate=0.001,
    nb_iter=50,
    inc_rate=1.2,
    dec_rate=0.5,
    verbose=True,
    use_pca=0.8
)

classifier.train_named(traindata, trainlabels)
predicted = classifier.predict_named(testdata)
print classifier.top_accuracy_named(testdata, testlabels)
