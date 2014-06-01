#include <float.h>

/**
 * Implementation of the generalized distance transform generalized to
 * quadratic distances. Implemented in C as I couldn't get an efficient
 * in Python or Cython.
 */
void gdt1D(float *d, float *f, int n, float *df) {
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
    qmvk = q - v[k];
    df[q] = d[0] * qmvk + d[1] * qmvk * qmvk + f[v[k]];
  }
}
