#include "cuda_runtime.h"

__global__ void eig(const float* M, float* V, float* L, const int n, bool useIterative);
__global__ void eigVal(const float* M, float* L, const int n);

__global__ void estimateNormals(const float* pts1, float* nor1,const int n);
