""" Cross validation and related ensemble methods.
"""
import numpy as np
import itertools
from random import shuffle
from classifier import ClassifierMixin

def k_fold_split(samples, labels, k):
    """ Splits the dataset into k subsets of roughly the same size,
        keeping the same proportion of samples for each class into
        each subset.

        Arguments:
            samples
                list of samples to split into k folds.
            labels
                numpy array of integer class labels associated to the samples.
            k
                number of subsets to output.
        Return:
            sample_subs, label_subs where both are k length lists containing
            sample subsets and label subsets respectively.
    """
    list_labels = not isinstance(labels, np.ndarray)
    # Group samples per class.
    classes = set(labels) if list_labels else set(labels.tolist())
    nb_samples = len(samples)
    samples_per_class = {}

    for i in classes:
        samples_per_class[i] = []

    for i in range(nb_samples):
        samples_per_class[labels[i]].append(samples[i])

    # Construct the folds by splitting each class into k
    # subsets roughly evenly.
    fold_samples = []
    fold_labels = []

    for i in range(k):
        fold_samples.append([])
        fold_labels.append([])
    for i in samples_per_class:
        nb_class_samples = len(samples_per_class[i])
        # shuffle the samples for the class.
        shuffle(samples_per_class[i])
        thresh = np.linspace(0, nb_class_samples, k+1).round().astype(np.int32)
        for j in range(k):
            fold_samples[j] += samples_per_class[i][thresh[j]:thresh[j+1]]
            fold_labels[j] += [i] * (thresh[j+1] - thresh[j])

    out_labels = fold_labels if list_labels else map(lambda l: np.array(l, np.int32), fold_labels)
    return (fold_samples, out_labels)

class BaseCVClassifier:
    """ Trains a classifier using k-fold cross validation and grid search to
        perform model selection, then keep the k trained models with best
        error rate for averaged predictino.
    """
    def __init__(self, classifier_class, k=4, verbose=False, args={}):
        self.classifier_class = classifier_class
        self.args = args
        self.k = k
        self.verbose = verbose
        
    def train(self, samples, labels):
        """ Trains the classifier with all possible combination
            of parameters, selecting the parameters performing best on
            a cross validation, then running training one final time on
            the entire training set using these parameters.

        Arguments:
            samples
                list of training samples of arbitrary datatype.
            labels
                numpy vector of labels in the {0,...,c-1} set where
                c is the number of classes.
        """
        assert len(samples) == labels.size

        fold_samples, fold_labels = k_fold_split(samples, labels, self.k)
        assert sum(map(len, fold_samples)) == len(samples)
        # Iterate over all combinations of parameters, find the best.
        nb_samples = len(samples)
        names = [name for name in self.args]
        params = [self.args[name] for name in names]
        iterator = itertools.product(*params)
        cv_results = []
        best_err_rate = 1.
        best_params = None
        best_models = None

        for arguments in iterator:
            argdict = {}
            for i in range(len(names)):
                argdict[names[i]] = arguments[i]
            # Measure the error rate using cross validation.
            nb_incorrect = 0
            
            if self.verbose:
                print "Running cross validation for " + repr(argdict)

            classifier = self.classifier_class(**argdict)
            for i in range(self.k):
                print "Fold " + repr(i + 1) + " out of " + repr(self.k)
                # Set up train and test set for the fold.
                testlabels = fold_labels[i]
                testsamples = fold_samples[i]
                trainlabels = np.concatenate(fold_labels[0:i] + fold_labels[i+1:], axis=0)
                trainsamples = reduce(lambda l1, l2: l1 + l2,
                                      fold_samples[0:i] + fold_samples[i+1:])
                print repr(nb_samples) + " total samples."
                print repr(len(testsamples)) + " test samples."
                print repr(len(trainsamples)) + " training samples."
                # Train the classifier.
                classifier.train(trainsamples, trainlabels)
                # Predict the test samples's labels.
                predicted = classifier.predict(testsamples)
                # Measure the error rate.
                for i in range(len(testlabels)):
                    if testlabels[i] != predicted[i]:
                        nb_incorrect += 1
            err_rate = float(nb_incorrect) / nb_samples
            if self.verbose:
                print "Finished cross validation for " + repr(argdict)
                print "Error rate: " + repr(err_rate)
            if err_rate < best_err_rate:
                best_params = argdict
                best_err_rate = err_rate

        # Finally, run training for the last time using the best 
        # performing set of parameters on the entire training set.
        if self.verbose:
            print "Best parameters found: " + repr(best_params)
            print "Error rate: " + repr(best_err_rate)
        self.classifier = self.classifier_class(**best_params)
        self.classifier.train(samples, labels)

    def predict_proba(self, samples):
        return self.classifier.predict_proba(samples)

    def predict(self, samples):
        return self.classifier.predict(samples)

class CVClassifier(BaseCVClassifier, ClassifierMixin):
    pass

