#include <float.h>

/**
 * Implementation of the generalized distance transform, generalized to
 * quadratic distances. Implemented in C as I couldn't get an efficient
 * implementation in Python or Cython.
 *
 * @param d coefficient for the distance function d(p,q) = d[0](p-q) + d[1](p-q)^2
 * @param f function to compute the distance transform of as an array.
 * @param n size of array f.
 * @param df output distance transform of f. Should be allocated to n elements prior
 *           to call.
 * @param arg output indexes corresponding to the argmax version of the gdt. Should be
 *            allocated to n elements prior to call.
 */
void gdt1D(float *d, float *f, int n, float *df, int *arg) {
  /*
   * Please refer to Felzenszwalb, Huttenlocher, 2004 for a detailed
   * pseudo code of the algorithm - which the code mirrors, except for
   * a few exceptions.
   */
  int k = 0;
  int v[n];
  float z[n * 2];
  float s;
  int qmvk;
  int q;
  v[0] = 0;
  z[0] = FLT_MIN;
  z[1] = FLT_MAX;
  
  for (q = 1; q < n; q++) {
    while (1) {
      /* Intersection s generalized to arbitrary parabolas (d[1] nonnegative). 
       * Follows from elementary algebra.
       */
      s = (d[0] * (v[k] - q) + d[1] * (q*q - v[k]*v[k]) + f[q] - f[v[k]])
	/ (2*d[1]*(q - v[k]));
      if (s > z[k]) {
	break;
      }
      k--;
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k+1] = FLT_MAX;
  }

  k = 0;
  for (q = 0; q < n; q++) {
    while (z[k+1] < q) {
      k++;
    }
    /* Compared to the original paper, swapped in the new distance definition. */
    qmvk = q - v[k];
    df[q] = d[0] * qmvk + d[1] * qmvk * qmvk + f[v[k]];
    /* Store the index of the actual max in the arg vector. Necessary for efficient
     * displacement lookup in the DPM matching algorithm. */
    arg[q] = k;
  }
}
