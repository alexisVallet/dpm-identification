#include <float.h>
#include <stdio.h>

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
  int vk, fvk, fq;
  v[0] = 0;
  z[0] = FLT_MIN;
  z[1] = FLT_MAX;
  
  for (q = 1; q < n; q++) {
    while (1) {
      /* Intersection s generalized to arbitrary parabolas (d[1] nonnegative). 
       * Follows from elementary algebra.
       */
      vk = v[k];
      fvk = f[vk];
      fq = f[q];
      s = (d[0] * (vk - q) + d[1] * (q*q - vk*vk) + fq - fvk)
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
    arg[q] = v[k];
  }
}

static int toRowMajor(int i, int j, int cols) {
  return i * cols + j;
}

static void fromRowMajor(int flat, int *i, int *j, int cols) {
  *i = flat / cols;
  *j = flat % cols;
}

static int toColMajor(int i, int j, int rows) {
  return i + j * rows;
}

static void fromColMajor(int flat, int *i, int *j, int rows) {
  *j = flat / rows;
  *i = flat % rows;
}

/**
 * Matrix transpose code from http://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c .
 */
void tran(float *src, float *dst, const int N, const int M) {
  int n, i, j;

  for(n = 0; n<N*M; n++) {
    i = n/N;
    j = n%N;
    dst[n] = src[M*j + i];
  }
}

/**
 * Compute the 2D generalized distance transform of a function. All output arrays should
 * be allocated prior to the call.
 *
 * @param d 4 elements array indicating coefficients for the quadratic distance.
 * @param f function to compute the distance transform of, rows*cols row-major matrix.
 * @param rows the number of rows on the grid.
 * @param cols the number of columns on the grid.
 * @param df output rows*cols row-major matrix for the distance transform of f.
 * @param argi output row indexes for the argmax version of the problem. rows*cols
 *             row-major matrix.
 * @param argj output column indexes for the argmax version of the problem. rows*cols
 *             row-major matrix.
 */
void gdt2D(float *d, float *f, int rows, int cols, 
	   float *df, int *argi, int *argj) {
  // apply the 1D algorithm on each row
  int i;
  int j;
  float dx[2] = {d[0], d[2]};
  float dy[2] = {d[1], d[3]};
  int offset;
  float df2[rows * cols];
  int tmpi, tmpj;

  printf("Computing on rows...\n");
  for (i = 0; i < rows; i++) {
    offset = toRowMajor(i,0,cols);
    //    printf("offset=%d, total size=%d\n", offset, rows * cols);
    gdt1D(dy, f + offset, cols, df + offset, argi + offset);
  }

  printf("Transposing...\n");
  // then on each column of the result. For this we transpose it, for memory locality.
  tran(df, df2, rows, cols);
  
  printf("Computing on columns...\n");
  for (i = 0; i < cols; i++) {
    offset = toColMajor(0, i, rows);
    gdt1D(dx, df2 + offset, rows, df2 + offset, argj + offset);
  }

  printf("Transposing again...\n");
  // transpose the result again
  tran(df2, df, cols, rows);

  printf("Computing indices...\n");
  // compute the indices for the arg arrays
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      fromRowMajor(argj[toRowMajor(i,j,cols)], &tmpi, &tmpj, cols);
      fromColMajor(argi[toColMajor(tmpi,tmpj,rows)], &tmpi, &tmpj, rows);
      argi[toRowMajor(i,j,cols)] = tmpi;
      argj[toRowMajor(i,j,cols)] = tmpj;
    }
  }
}
