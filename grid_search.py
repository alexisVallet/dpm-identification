""" Runs grid search on a classifier as part of the training procedure,
    choosing the parameters performing best on a K-fold cross validation.
"""
import numpy as np
import itertools
from random import shuffle

class GridSearchMixin:
    """ Mixin for classifiers adding grid search functionality, as well
        as named labels. The class should implement the following:
        - train(samples, int_labels)
        - predict(samples) -> int_labels
        - verbose -> boolean
        - a constructor which takes the arguments to perform grid search
          on, and fully resets the classifier (ie. one can call __init__
          multiple times without corrupting the state).
    """
    def predict_named(self, samples):
        int_labels = self.predict(samples)
        return map(
            lambda i: self.int_to_label[i],
            self.predict(samples).tolist()
        )

    def train_named(self, samples, labels):
        self.int_to_label = list(set(labels))
        label_to_int = {}
        
        for i in range(len(self.int_to_label)):
            label_to_int[self.int_to_label[i]] = i
        
        int_labels = np.array(
            map(lambda l: label_to_int[l], labels),
            dtype=np.int32
        )
        self.train(samples, int_labels)

    def train_gs_named(self, samples, labels, k, **args):
        """ Trains a classifier with grid search using named labels.
        """
        self.int_to_label = list(set(labels))
        label_to_int = {}
        
        for i in range(len(self.int_to_label)):
            label_to_int[self.int_to_label[i]] = i
        
        int_labels = np.array(
            map(lambda l: label_to_int[l], labels),
            dtype=np.int32
        )
        self.train_gs(samples, int_labels, k, **args)

    def _train_gs(self, shflsamples, shfllabels, k, **args):
        # Iterate over all combinations of parameters, find the best.
        nb_samples = len(shflsamples)
        thresh = np.round(np.linspace(0, nb_samples, k + 1)).astype(np.int32)
        names = [name for name in args]
        params = [args[name] for name in names]
        iterator = itertools.product(*params)
        cv_results = []
        best_err_rate = 1.
        best_params = None

        for arguments in iterator:
            argdict = {}
            for i in range(len(names)):
                argdict[names[i]] = arguments[i]
            # Measure the error rate using cross validation.
            nb_incorrect = 0
            
            if self.verbose:
                print "Running cross validation for " + repr(argdict)

            for i in range(k):
                print "Fold " + repr(i + 1) + " out of " + repr(k)
                # Set up train and test set for the fold.
                testlabels = shfllabels[thresh[i]:thresh[i+1]]
                testsamples = shflsamples[thresh[i]:thresh[i+1]]
                trainlabels = np.concatenate((
                    shfllabels[0:thresh[i]],
                    shfllabels[thresh[i+1]:]
                ), axis=0)
                trainsamples = (
                    shflsamples[0:thresh[i]] + 
                    shflsamples[thresh[i+1]:]
                )
                # Train the classifier.
                self.__init__(**argdict)
                self.train(trainsamples, trainlabels)
                # Predict the test samples's labels.
                predicted = self.predict(testsamples)
                # Measure the error rate.
                for i in range(len(testlabels)):
                    if testlabels[i] != predicted[i]:
                        nb_incorrect += 1
            err_rate = float(nb_incorrect) / nb_samples
            if self.verbose:
                print "Finished cross validation for " + repr(argdict)
                print "Error rate: " + repr(err_rate)
            cv_results.append((argdict, err_rate))
            if err_rate < best_err_rate:
                best_params = argdict
                best_err_rate = err_rate
        return (best_params, best_err_rate)

    def shuffle_data(self, samples, labels):
        nb_samples = len(samples)
        # Randomly choose folds.
        shuffledidxs = np.random.permutation(nb_samples)
        shfllabels = labels[shuffledidxs]
        shflsamples = []
        for i in range(nb_samples):
            shflsamples.append(samples[shuffledidxs[i]])

        return (shflsamples, shfllabels)

    def train_gs(self, samples, labels, k, **args):
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
        shflsamples, shfllabels = self.shuffle_data(samples, labels)
        
        best_params, best_err_rate = self._train_gs(
            shflsamples,
            shfllabels,
            k,
            **args
        )

        # Finally, run training for the last time using the best 
        # performing set of parameters on the entire training set.
        if self.verbose:
            print "Best parameters found: " + repr(best_params)
            print "Now training on the entire training set..."
        self.__init__(best_params)
        self.train(samples, labels)
