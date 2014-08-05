""" Combines multiple probabilistic binary classifiers by averageing their
    outputs.
"""
import numpy as np

class Combination:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def train(self, positives, negatives):
        for classifier in self.classifiers:
            classifier.train(positives, negatives)
    
    def predict_proba(self, samples):
        sumprobas = np.zeros([len(samples)])
        for classifier in self.classifiers:
            sumprobas += classifier.predict_proba(samples)

        return sumprobas / len(self.classifiers)
