#include "aabb.h"

#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>

#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <execution> 


// typedef StructAlignment(32) std::array<_simd, 6> SimdObject;

bool does_collide(const Aabb& a, const Aabb& b)
{
    return 
    //    a.max[0] >= b.min[0] && a.min[0] <= b.max[0] && //ignore x axis
            a.max[1] >= b.min[1] && a.min[1] <= b.max[1] &&
            a.max[2] >= b.min[2] && a.min[2] <= b.max[2];
}

bool covertex(const int* a, const int* b) {
    
    return a[0] == b[0] || a[0] == b[1] || a[0] == b[2] || 
        a[1] == b[0] || a[1] == b[1] || a[1] == b[2] || 
        a[2] == b[0] || a[1] == b[1] || a[2] == b[2];
}

void add_overlap(const int& xid, const int& yid, atomic_int& count, int * overlaps, int G)
{
    // int i = atomicAdd(count, 1); //how to do this

    // do x+=y and return the old value of x
    int i = count++;

    if (i < G)
    {
        overlaps[2*i] = xid;
        overlaps[2*i+1] = xid;
    } 
}

// https://stackoverflow.com/questions/3909272/sorting-two-corresponding-arrays
class sort_indices
{
   private:
     Aabb* mparr;
   public:
     sort_indices(Aabb* parr) : mparr(parr) {}
     bool operator()(int i, int j) const { return (mparr[i].min[0] < mparr[j].min[0]);}
};

struct sort_boxes //sort_aabb_x
{
    bool operator()(const Aabb &a, const Aabb &b) const {
        return (a.min[0] < b.min[0]);}
};

// int arr[5]={4,1,3,6,2}
// string arr1[5]={"a1","b1","c1","d1","e1"};
// int indices[5]={0,1,2,3,4};

void sort_along_xaxis(Aabb * boxes, int * box_indices, int N)
{
    // sort box indices by boxes minx val
    sort(execution::par_unseq, box_indices, box_indices + N, sort_indices(boxes));
    // sort boxes by minx val
    sort(execution::par_unseq, boxes, boxes + N, sort_boxes());
}

void sweep(const Aabb * boxes, const int * box_indices, atomic_int& count, int * overlaps, int N, int guess)
{
    // ask about changing number of boxes per thread !!!
    // tbb::parallel_for(0, queries.size(), 1, [&](int i){
    // // body of the for loop using index i
    //     }); 

    tbb::parallel_for( tbb::blocked_range<int>(0, N),
                       [&](tbb::blocked_range<int> r)
                        {
                             for (int i=r.begin(); i<r.end(); i++)
                           {
                               const Aabb a = boxes[i];

                               int inc = i + 1;
                               Aabb b = boxes[inc];

                               while (a.max[0]  >= b.min[0])
                               {
                                   if (
                                        does_collide(a,b) &&
                                        !covertex(a.vertexIds, b.vertexIds)
                                            )
                                        {
                                            add_overlap(box_indices[i], box_indices[inc], count, overlaps, guess);
                                        }
                               }

                            //    Eigen::Matrix<double, 8, 3> V=queries[i];
                           }
                        }
    );


    // box a = boxes[threadId]
    // t = threadId + 1;
    // box b = boxes[t]
    
    // while a->max.x >= b->min.x
    // does_collide &&
    // !covertex
    // add_overlap
    // b = boxes[t + 1]
}


void run_sweep_cpu(
    Aabb* boxes, 
    int N, int numBoxes, 
    vector<unsigned long>& finOverlaps)
{
    // sort boxes by xaxis in parallel
    // we will need an index vector
    int * box_indices = new int[N];
    for (size_t i=0;i<N;i++) {box_indices[i]=i;}
    sort_along_xaxis(boxes, box_indices, N);

    int guess = 0;
    int * overlaps = new int(2*guess);
    
    atomic_int count(0);
    // count[0] = 0;

    sweep(boxes, box_indices, count, overlaps, N, guess);
    if (count > guess) //we went over
    {
        guess = count;
        delete[] overlaps;  //probably dont need
        overlaps = new int(2*guess);
        count = 0;
        sweep(boxes, box_indices, count, overlaps, N, guess);
    }

    for (size_t i=0; i < count; i++)
    {
        // finOverlaps.push_back(overlaps[i].x);
        // finOverlaps.push_back(overlaps[i].y);
        
        // need to fetch where box is from index first
        const Aabb& a = boxes[box_indices[overlaps[2*i]]];
        const Aabb& b = boxes[box_indices[overlaps[2*i+1]]];
        if (a.type == Simplex::VERTEX && b.type == Simplex::FACE)
        {
            finOverlaps.push_back(a.ref_id);
            finOverlaps.push_back(b.ref_id);
        }
        else if (a.type == Simplex::FACE && b.type == Simplex::VERTEX)
        {
            finOverlaps.push_back(b.ref_id);
            finOverlaps.push_back(a.ref_id);
        }
        else if (a.type == Simplex::EDGE && b.type == Simplex::EDGE)
        {
            finOverlaps.push_back(min(a.ref_id, b.ref_id));
            finOverlaps.push_back(max(a.ref_id, b.ref_id));
        }
    }
}

    // #pragma omp declare reduction (merge : std::vector<long> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    // #pragma omp parallel for num_threads(num_threads),reduction(+:m_narrowPhase), reduction(merge:narrowPhaseValues)

    