#include "simple.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include "cuda_runtime.h"

void simple(){
   // generate 16M random numbers on the host
   thrust::host_vector<int> h_vec(1 << 24);
   thrust::generate(h_vec.begin(), h_vec.end(), rand);
    
   // transfer data to the device
   thrust::device_vector<int> d_vec = h_vec; // sort data on the device
   thrust::sort(d_vec.begin(), d_vec.end()); // transfer data back to host
   thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}

namespace LoadingSaving{

pair< vector<float3>, vector<float3> > loadPointsAndNormals(std::string filename){
    std::ifstream file(filename.c_str(),std::ifstream::in);
    if( file.fail() == true )
    {
        cerr << filename << " could not be opened" << endl;
    }

    vector<float3> pts,nor;

    while(file){
    	float3 pt,no;
    	file >> pt.x >> pt.y >> pt.z >> no.x >> no.y >> no.z;
    	pts.push_back(pt);
        nor.push_back(no);
    }

    return make_pair(pts,nor);
}

}
