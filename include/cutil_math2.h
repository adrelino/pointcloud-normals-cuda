#ifndef CUTIL_MATH2_H
#define CUTIL_MATH2_H

#include "cuda_runtime.h"

#include "cuda_runtime.h"

#include <numeric>
#include <iostream>
#include <fstream>

inline std::ostream& operator<<(std::ostream& os, const float3& v) {
  os << v.x << " " << v.y << " " << v.z << std::endl;
  return os;
}

// outer product of 3x1 vector with itself, add to 3x3 matrix
inline __host__ __device__ void outerAdd(const float3 v, float *m, const float fac = 1.0f) {  //m column major
  m[0] += v.x * v.x * fac;
  m[4] += v.y * v.y * fac;
  m[8] += v.z * v.z * fac;

  float m01 = v.x*v.y * fac;
  m[1] +=m01;
  m[3] +=m01;

  float m02 = v.x*v.z * fac;
  m[2] +=m02;
  m[6] +=m02;

  float m12 = v.y*v.z * fac;
  m[5] +=m12;
  m[7] +=m12;
}

inline __host__ __device__ void mul(float *m, const float fac = 1.0f) {  //m column major
  m[0] *= fac;
  m[1] *= fac;
  m[2] *= fac;
  m[3] *= fac;
  m[4] *= fac;
  m[5] *= fac;
  m[6] *= fac;
  m[7] *= fac;
  m[8] *= fac;
}

#endif // CUTIL_MATH2_H
