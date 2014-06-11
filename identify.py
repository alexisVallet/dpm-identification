""" Identification algorithm for the binary classification problem.
Essentially based on the DPM matching algorithm by Felzenszwalb and
Girshick.
"""
import matching

def identify(pyramid, model):
    """ Returns the score of a mixture of dpms on a feature pyramid, and
        optionally the information about the corresponding object hypothesis.

    Arguments:
        pyramid    pyramid to match the model on.
        model      model to match on the feature pyramid.

    Returns:
        The score of the best match for the model on the pyramid.
    """
    
