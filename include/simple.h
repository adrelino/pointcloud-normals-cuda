#include <iostream>
#include <vector>
#include <string>
#include "cuda_runtime.h"


void simple();

namespace LoadingSaving{
	using namespace std;

	pair< vector<float3>, vector<float3> > loadPointsAndNormals(string filename);
}
