""" Cython module implementing the generalized distance transform by Felzenszwalb and
    Huttenlocher, 2004. With a slight twist: it implements it for the slightly more
    general distance d(p,q) = a(p-q) + b(p-q)^2 where b is positive. For use with my
    variation of the DPM learning algorithm for character identification, and for
    prediction as well.
"""
import numpy as np

def distancetransform1D(d, f):
    """ Computes the 1D distance transform of a function f with distance specified
        by dist(p, q) = d . (p-q, (p-q)^2), i.e. a quadratic function of the
        displacement between p and q, with the constraint that d[1] must be positive.
    
    Args:
        d    2 elements numpy vector specifying the distance function.
        f    function to compute the distance transform of. 1d numpy array where f[i]
             is the value of the function at i.
    Returns:
       an array of the same size as f specifying its distance transform.
    """
    if d(1) <= 0:
        raise ValueError("The quadratic coefficient must be positive!")
    # The code is pretty much a copy paste of the pseudo code I have on paper,
    # sorry for the bad readability. Please see the paper by Felzenszwalb and
    # Huttenlocher from 2004 for a detailed explanation. The only difference is the
    # update of s and the final building of the result to account for the slightly
    # more general distance function.
    n = f.shape
    k = 0
    v = np.empty(n)
    v[0] = 0
    z = np.empty(n * 2)
    z[0] = -np.inf
    z[1] = np.inf
    
    for q in range(1, n):
        while True:
            qvec = np.array([v[k] - q, q**2 - v[k]**2])
            # Modified from the paper. The intersection of the 2 parabolas can
            # easily be determined by elementary algebra.
            s = (np.vdot(d, qvec) + f[q] - f[v[k]]) / (2*d[1]*(q - v[k]))
            if s > z[k]:
                break;
            k = k - 1
        k = k + 1
        v[k] = q
        z[k] = s
        z[k+1] = np.inf

    k = 0
    df = np.empty(n)

    for q in range(0,n):
        while z[k+1] < q:
            k = k + 1
        qvec = np.array([q - v[k], (q - v[k])**2])
        # Once again slightly different, this is simply the definition of the
        # distance transform with my new distance swapped in.
        df[q] = np.vdot(d, qvec) + f(v[k])
    return df

        
