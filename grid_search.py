""" Runs grid search on a classifier as part of the training procedure,
    choosing the parameters performing best on a validation set.
    (i.e. minimizing the classification error rate)
"""
import numpy as np
import itertools

class GridSearch:
    def __init__(self, classifier_init, args, validation_size=0.2,
                 verbose=False):
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
            validation_size
                float between 0 and 1, specifies the fraction of the
                training set to use as a validation set.
            verbose
                set to true for printing information messages at regular
                intervals.
        """
        assert 0 < validation_size < 1
        self.classifier_init = classifier_init
        self.args = args
        self.validation_size = validation_size
        self.verbose = verbose

    def train(self, samples, labels):
        """ Trains the inner classifier with all possible combination
            of parameters, selecting the parameters performing best on
            a validation set, then running training one final time on
            the entire training set using these parameters.
        """
        assert len(samples) == len(labels)
        # Randomly choose a validation set.
        shuffledidxs = np.random.permutation(len(samples))
        thresh = int(float(len(samples)) * self.validation_size)
        trainidxs = shuffledidxs[0:thresh]
        validationidxs = shuffledidxs[thresh:]
        validation_samples = samples[validationidxs]
        validation_labels = labels[validationidxs]
        train_samples = samples[trainidxs]
        train_labels = labels[trainidxs]
        # Iterate over all combinations of parameters, find the best.
        names = [name for name in self.args]
        params = [self.args[name] for name in names]
        iterator = itertools.product(*params)
        

        for arguments in iterator:
            argdict = {}
            for i in range(len(names)):
                argdict[names[i]] = arguments[i]
            classifier = self.classifier_init(argdict)
            classifier.train(train_samples, train_labels)
            # Measure the error rate.
        
