#include <math.h>
#include "cutil_math.h"
#include "cutil_math2.h"
#include "cov.h"

__device__ float3 d_mean(const float3* pts, const int n){
	float3 m = make_float3(0,0,0);
	for (int i = 0; i < n; ++i)
	{
		m += pts[i];
	}
	m /= (n+0.0f);
	return m;
}

__device__ void d_cov(const float3* pts, float* C, const int n){
	const float3 m = d_mean(pts,n);

	for (int i = 0; i < n; ++i)
	{
        float3 diff = pts[i]-m;
		outerAdd(diff,C);
	}

	float fac=1.0f/(n-1.0f);

	for (int i = 0; i < 9; ++i)
	{
		C[i] *=fac;
	}
}

__global__ void mean(const float* pts, float* m, const int n){
	const float3 *pts1= (float3*) pts;
	//float3 *m3 = (float3*) m;
	float3 mm = d_mean(pts1,n);
	m[0]=mm.x;
	m[1]=mm.y;
	m[2]=mm.z;
}

//	float C[9]={0,0,0, 0,0,0, 0,0,0};
__global__ void cov(const float* pts, float* C, const int n){
	const float3 *pts1= (float3*) pts;
    d_cov(pts1,C,n);
}
