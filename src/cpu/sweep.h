#pragma once

#include "aabb.h"
#include <vector>

using namespace std;

namespace ccdcpu {

void run_sweep_cpu(
    vector<Aabb>& boxes, 
    int N, int numBoxes, 
    vector<pair<long,long>>& finOverlaps);

} //namespace