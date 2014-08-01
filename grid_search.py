""" Runs grid search on a classifier as part of the training procedure,
    choosing the parameters performing best on a K-fold cross validation.
"""
import numpy as np
import itertools
from random import shuffle

class GridSearch:
    def __init__(self, classifier_init, args, k=3, verbose=False):
        """ Initializes the grid search classifier with a specific
            initialization function, and a dictionary of arguments
            to search over.

        Arguments:
            classifier_init
                function taking a dictionary of arguments as parameters,
                and returning an object implementing the
                train(samples, labels) method as well as the 
                predict(samples) method. Should be picklable.
            args
                dictionary specifying the space of arguments to search
                over. i.e. if args = {C: [1, 0.1], alpha: [0.1, 0.01]}
                then classifier init will be called with 
                {C: 1, alpha: 0.1} then {C: 1, alpha: 0.01} etc.
            k
                number of folds in the cross validation.
            verbose
                set to true for printing information messages at regular
                intervals.
        """
        self.classifier_init = classifier_init
        self.args = args
        self.k = k
        self.verbose = verbose

    def train(self, samples, labels):
        """ Trains the inner classifier with all possible combination
            of parameters, selecting the parameters performing best on
            a cross validation, then running training one final time on
            the entire training set using these parameters.
        """
        assert len(samples) == len(labels)
        nb_samples = len(samples)
        # Randomly choose folds.
        shuffled = zip(samples, labels)
        shuffle(shuffled)
        thresh = np.round(np.linspace(0, nb_samples, self.k + 1)).astype(np.int32)
        # Iterate over all combinations of parameters, find the best.
        names = [name for name in self.args]
        params = [self.args[name] for name in names]
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

            for i in range(self.k):
                print "Fold " + repr(i + 1) + " out of " + repr(self.k)
                # Set up train and test set for the fold.
                test = shuffled[thresh[i]:thresh[i+1]]
                testsamples, testlabels = zip(*test)
                train = shuffled[0:thresh[i]] + shuffled[thresh[i+1]:]
                trainsamples, trainlabels = zip(*train)
                # Train the classifier.
                classifier = self.classifier_init(argdict)
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
            cv_results.append((argdict, err_rate))
            if err_rate < best_err_rate:
                best_params = argdict
                best_err_rate = err_rate
        # Finally, run training for the last time using the best performing
        # set of parameters on the entire training set.
        if self.verbose:
            print "Best parameters found: " + repr(best_params)
            print "Now training on the entire training set..."
        self.classifier = self.classifier_init(best_params)
        self.classifier.train(samples, labels)

    def predict(self, samples):
        return self.classifier.predict(samples)
