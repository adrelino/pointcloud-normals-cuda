// @note CUDA6 supports clang, but DO NOT support libc++, so you need to use libstdc++!!!
//
// /usr/local/cuda/bin/nvcc simple.cu -c -o simple.cu.o -ccbin /usr/local/bin/clang++ -Xcompiler -stdlib=libstdc++
// /usr/bin/ar cr libcuda.a simple.cu.o
// /usr/local/bin/clang++ main.cpp libcuda.a /usr/local/cuda/lib/libcudart.dylib -Wl,-rpath -Wl,/usr/local/cuda/lib -Wl,-rpath,/usr/local/cuda/lib

#include "simple.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "cutil_math2.h"
using namespace std;

int main(void){ 
    simple();
    pair< vector<float3>, vector<float3> > model = LoadingSaving::loadPointsAndNormals("bunny/model.xyz");

    int n = model.first.size();
    cout<<n<<" pts"<<endl;

    int nrBytes = n*sizeof(float3);
    float3* h_pts = &model.first[0];
    float3* d_pts;
    cudaMalloc(&d_pts,nrBytes);
    cudaMemcpy(d_pts,h_pts,nrBytes,cudaMemcpyHostToDevice);

    float3* h_pts2 = new float3[n];
    cudaMemcpy(h_pts2,d_pts,nrBytes,cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++){
        cout<<h_pts2[i]-h_pts[i]<<endl;
    }

    return 0; 
}
