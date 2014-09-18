import numpy as np
from dpm_classifier import DPMClassifier
from cross_validation import k_fold_split, CVClassifier
from ioutils import load_data
from features import Combine, HoG, BGRHist

# Parameters.
outer_k = 5
inner_k = 4
args = {
    'feature': [Combine(HoG(9,1), BGRHist((4,4,4), 0))],
    'mindimdiv': [10],
    'nbparts': [2, 4, 6, 8],
    'C': [0.1],
    'learning_rate': [0.001],
    'nb_gd_iter': [25, 50, 75, 100],
    'inc_rate': [1.2],
    'dec_rate': [0.5],
    'verbose': [True],
    'use_pca': [None]
}

# Loading the full dataset.
print "Loading data..."
data = load_data(
    'data/images/source/',
    'data/json/boundingboxes'
)

nb_classes = len([l for l in data])

# Convert to a list of images and a list of labels.
samples = []
labels = []

for l in data:
    for s in data[l]:
        samples.append(s)
        labels.append(l)

nb_samples = len(samples)
        
# Split the entire data into 5 folds.
fold_samples, fold_labels = k_fold_split(samples, labels, outer_k)

print "Running 5-fold outer cross-validation..."
# We'll assume that the folds have the exact same size, and compute
# the average top-1 to 20 accuracies across them.
top_accuracy_avg = np.zeros([nb_classes]) 
for i in range(outer_k):
    print "Outer fold " + repr(i + 1) + " out of " + repr(outer_k)
    testsamples = fold_samples[i]
    testlabels = fold_labels[i]
    concat = lambda l: reduce(lambda l1, l2: l1 + l2, l)
    trainsamples = concat(fold_samples[0:i] + fold_samples[i+1:])
    trainlabels = concat(fold_labels[0:i] + fold_labels[i+1:])
    print repr(len(trainsamples)) + " training samples."
    print repr(len(testsamples)) + " test samples."

    classifier = CVClassifier(
        DPMClassifier,
        k=inner_k,
        verbose=True,
        args=args
    )
    print "Training..."
    classifier.train_named(trainsamples, trainlabels)
    print "Predicting..."
    top_accuracy = classifier.top_accuracy_named(testsamples, testlabels)
    print "top accuracy for outer fold " + repr(i+1) + ":"
    print top_accuracy
    top_accuracy_avg += top_accuracy
print "Completed outer cross-validation."
top_accuracy_avg = top_accuracy_avg / outer_k
print "Average top accuracy across all folds:"
print top_accuracy_avg
