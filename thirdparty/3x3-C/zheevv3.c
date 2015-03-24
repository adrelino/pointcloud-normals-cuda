// ----------------------------------------------------------------------------
// Numerical diagonalization of 3x3 matrcies
// Copyright (C) 2006  Joachim Kopp
// ----------------------------------------------------------------------------
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
// ----------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <float.h>
#include "zheevc3.h"
#include "zheevv3.h"

// Macros
#define SQR(x)      ((x)*(x))                        // x^2 
#define SQR_ABS(x)  (SQR(creal(x)) + SQR(cimag(x)))  // |x|^2

// ----------------------------------------------------------------------------
int zheevv3(double complex A[3][3], double complex Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   zheevc3()
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
  double norm;          // Squared norm or inverse norm of current eigenvector
  double n0, n1;        // Norm of first and second columns of A
  double n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
  double thresh;        // Small number used as threshold for floating point comparisons
  double error;         // Estimated maximum roundoff error in some steps
  double wmax;          // The eigenvalue of maximum modulus
  double complex f;     // Intermediate storage
  double t;             // Intermediate storage
  int i, j;             // Loop counters
#endif

  // Calculate eigenvalues
  zheevc3(A, w);

#ifndef EVALS_ONLY
  wmax = fabs(w[0]);
  if ((t=fabs(w[1])) > wmax)
    wmax = t;
  if ((t=fabs(w[2])) > wmax)
    wmax = t;
  thresh = SQR(8.0 * DBL_EPSILON * wmax);

  // Prepare calculation of eigenvectors
  n0tmp   = SQR_ABS(A[0][1]) + SQR_ABS(A[0][2]);
  n1tmp   = SQR_ABS(A[0][1]) + SQR_ABS(A[1][2]);
  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*creal(A[1][1]);
  Q[1][1] = A[0][2]*conj(A[0][1]) - A[1][2]*creal(A[0][0]);
  Q[2][1] = SQR_ABS(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = conj( (A - w[0]).e1 x (A - w[0]).e2 )
  A[0][0] -= w[0];
  A[1][1] -= w[0];
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = creal(A[0][0])*creal(A[1][1]) - Q[2][1];
  norm    = SQR_ABS(Q[0][0]) + SQR_ABS(Q[1][0]) + SQR(creal(Q[2][0]));
  n0      = n0tmp + SQR(creal(A[0][0]));
  n1      = n1tmp + SQR(creal(A[1][1]));
  error   = n0 * n1;
  
  if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
  {
    Q[0][0] = 1.0;
    Q[1][0] = 0.0;
    Q[2][0] = 0.0;
  }
  else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
  {
    Q[0][0] = 0.0;
    Q[1][0] = 1.0;
    Q[2][0] = 0.0;
  }
  else if (norm < SQR(64.0 * DBL_EPSILON) * error)
  {                         // If angle between A[0] and A[1] is too small, don't use
    t = SQR_ABS(A[0][1]);   // cross product, but calculate v ~ (1, -A0/A1, 0)
    f = -A[0][0] / A[0][1];
    if (SQR_ABS(A[1][1]) > t)
    {
      t = SQR_ABS(A[1][1]);
      f = -conj(A[0][1]) / A[1][1];
    }
    if (SQR_ABS(A[1][2]) > t)
      f = -conj(A[0][2] / A[1][2]);
    norm    = 1.0/sqrt(1 + SQR_ABS(f));
    Q[0][0] = norm;
    Q[1][0] = f * norm;
    Q[2][0] = 0.0;
  }
  else                      // This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }

  
  // Prepare calculation of second eigenvector
  t = w[0] - w[1];
  if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
  {
    // For non-degenerate eigenvalue, calculate second eigenvector by the formula
    //   v[1] = conj( (A - w[1]).e1 x (A - w[1]).e2 )
    A[0][0] += t;
    A[1][1] += t;
    Q[0][1]  = Q[0][1] + A[0][2]*w[1];
    Q[1][1]  = Q[1][1] + A[1][2]*w[1];
    Q[2][1]  = creal(A[0][0])*creal(A[1][1]) - creal(Q[2][1]);
    norm     = SQR_ABS(Q[0][1]) + SQR_ABS(Q[1][1]) + SQR(creal(Q[2][1]));
    n0       = n0tmp + SQR(creal(A[0][0]));
    n1       = n1tmp + SQR(creal(A[1][1]));
    error    = n0 * n1;
 
    if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
    {
      Q[0][1] = 1.0;
      Q[1][1] = 0.0;
      Q[2][1] = 0.0;
    }
    else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
    {
      Q[0][1] = 0.0;
      Q[1][1] = 1.0;
      Q[2][1] = 0.0;
    }
    else if (norm < SQR(64.0 * DBL_EPSILON) * error)
    {                       // If angle between A[0] and A[1] is too small, don't use
      t = SQR_ABS(A[0][1]); // cross product, but calculate v ~ (1, -A0/A1, 0)
      f = -A[0][0] / A[0][1];
      if (SQR_ABS(A[1][1]) > t)
      {
        t = SQR_ABS(A[1][1]);
        f = -conj(A[0][1]) / A[1][1];
      }
      if (SQR_ABS(A[1][2]) > t)
        f = -conj(A[0][2] / A[1][2]);
      norm    = 1.0/sqrt(1 + SQR_ABS(f));
      Q[0][1] = norm;
      Q[1][1] = f * norm;
      Q[2][1] = 0.0;
    }
    else
    {
      norm = sqrt(1.0 / norm);
      for (j=0; j < 3; j++)
        Q[j][1] = Q[j][1] * norm;
    }
  }
  else
  {
    // For degenerate eigenvalue, calculate second eigenvector according to
    //   v[1] = conj( v[0] x (A - w[1]).e[i] )
    //   
    // This would really get to complicated if we could not assume all of A to
    // contain meaningful values.
    A[1][0]  = conj(A[0][1]);
    A[2][0]  = conj(A[0][2]);
    A[2][1]  = conj(A[1][2]);
    A[0][0] += w[0];
    A[1][1] += w[0];
    for (i=0; i < 3; i++)
    {
      A[i][i] -= w[1];
      n0       = SQR_ABS(A[0][i]) + SQR_ABS(A[1][i]) + SQR_ABS(A[2][i]);
      if (n0 > thresh)
      {
        Q[0][1]  = conj(Q[1][0]*A[2][i] - Q[2][0]*A[1][i]);
        Q[1][1]  = conj(Q[2][0]*A[0][i] - Q[0][0]*A[2][i]);
        Q[2][1]  = conj(Q[0][0]*A[1][i] - Q[1][0]*A[0][i]);
        norm     = SQR_ABS(Q[0][1]) + SQR_ABS(Q[1][1]) + SQR_ABS(Q[2][1]);
        if (norm > SQR(256.0 * DBL_EPSILON) * n0) // Accept cross product only if the angle between
        {                                         // the two vectors was not too small
          norm = sqrt(1.0 / norm);
          for (j=0; j < 3; j++)
            Q[j][1] = Q[j][1] * norm;
          break;
        }
      }
    }
    
    if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
    {
      for (j=0; j < 3; j++)
        if (creal(Q[j][0]) != 0.0  ||  cimag(Q[j][0]) != 0.0) // Find nonzero element of v[0] ...
        {                                                     // ... and swap it with the next one
          norm          = 1.0 / sqrt(SQR_ABS(Q[j][0]) + SQR_ABS(Q[(j+1)%3][0]));
          Q[j][1]       = conj(Q[(j+1)%3][0]) * norm;
          Q[(j+1)%3][1] = -conj(Q[j][0]) * norm;
          Q[(j+2)%3][1] = 0.0;
          break;
        }
    }
  }
      
  
  // Calculate third eigenvector according to
  //   v[2] = conj(v[0] x v[1])
  Q[0][2] = conj(Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1]);
  Q[1][2] = conj(Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1]);
  Q[2][2] = conj(Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1]);
#endif

  return 0;
}

