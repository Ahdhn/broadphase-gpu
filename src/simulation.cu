#include <gpubf/simulation.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#include "tbb/concurrent_vector.h"
#include <tbb/task_group.h>

#include <cmath>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void setup(int devId, int& smemSize, int& threads, int& nboxes);



void run_collision_counter(Aabb* boxes, int N) {

    // int N = 200000;
    // Aabb boxes[N];
    // for (int i = 0; i<N; i++)
    // {
    //     boxes[i] = Aabb(i);
    //     // printf("box %i created\n", boxes[i].id);
    // }

    // Allocate boxes to GPU 
    Aabb * d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb)*N, cudaMemcpyHostToDevice);

    // Allocate counter to GPU + set to 0 collisions
    int * d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    reset_counter<<<1,1>>>(d_counter);
    cudaDeviceSynchronize();

     int collisions;
    // cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // int bytes_mem_intrfce = 352 >> 3;
    // int mem_clock_rate = 1376 << 1;
    // float bandwidth_mem_theor = (mem_clock_rate * bytes_mem_intrfce) / pow(10, 3);

    // Set up timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Get number of collisions
    cudaEventRecord(start);
    count_collisions<<<1,1>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("(count_collisions<<<1,1>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.6f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);
    printf("Effective Bandwidth (GB/s): %.6f (GB/s)\n", 32*2/milliseconds/1e6);

    reset_counter<<<1,1>>>(d_counter);
    cudaEventRecord(start);
    count_collisions<<<1,1024>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n(count_collisions<<<1,1024>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.6f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);

    reset_counter<<<1,1>>>(d_counter);
    cudaEventRecord(start);
    count_collisions<<<2,1024>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n(count_collisions<<<2,1024>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.6f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);

    reset_counter<<<1,1>>>(d_counter);
    cudaEventRecord(start);
    count_collisions<<<56,1024>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n(count_collisions<<<56,1024>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.9f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);

    reset_counter<<<1,1>>>(d_counter);
    cudaEventRecord(start);
    count_collisions<<<256,1024>>>(d_boxes, d_counter, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n(count_collisions<<<256,1024>>>)\n");
    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Elapsed time: %.9f ms/c\n", milliseconds/collisions);
    printf("Collision: %i\n", collisions);
    return;
    // printf("%zu\n", sizeof(Aabb));


    // Retrieve count from GPU and print out
    // int counter;
    // cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("count: %d\n", counter);
    // return 0;
}

void run_scaling(const Aabb* boxes,  int N, int desiredBoxesPerThread, vector<unsigned long>& finOverlaps)
{

    int devId = 0;
    cudaSetDevice(devId);

    int smemSize;
    int threads;

    setup(devId, smemSize, threads, desiredBoxesPerThread);
    const int nBoxesPerThread = desiredBoxesPerThread ? desiredBoxesPerThread : smemSize / sizeof(Aabb) / (2*(BLOCK_PADDED));
    printf("Boxes per Thread: %i\n", nBoxesPerThread);

    finOverlaps.clear();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    // guess overlaps size
    int guess = 0;

    // Allocate boxes to GPU 
    Aabb * d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb)*N, cudaMemcpyHostToDevice);

    // Allocate counter to GPU + set to 0 collisions
    int * d_count;
    cudaMalloc((void**)&d_count, sizeof(int));
    reset_counter<<<1,1>>>(d_count);
    cudaDeviceSynchronize();

    //Count collisions
    count_collisions<<<1,1>>>(d_boxes, d_count, N); 
    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    reset_counter<<<1,1>>>(d_count);
    printf("Total collisions from counting: %i\n", count);



    int2 * d_overlaps;
    cudaMalloc((void**)&d_overlaps, sizeof(int2)*(guess));

    dim3 block(BLOCK_SIZE_1D,BLOCK_SIZE_1D);
    // dim3 grid ( (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D,  (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D );
    int grid_dim_1d = (N+BLOCK_SIZE_1D)/ BLOCK_SIZE_1D / nBoxesPerThread;
    dim3 grid( grid_dim_1d, grid_dim_1d );
    printf("Grid dim (1D): %i\n", grid_dim_1d);
    printf("Box size: %i\n", sizeof(Aabb));

    long long * d_queries;
    cudaMalloc((void**)&d_queries, sizeof(long long)*(1));
    reset_counter<<<1,1>>>(d_queries);

    printf("Shared mem alloc: %i B\n", nBoxesPerThread*2*(BLOCK_PADDED)*sizeof(Aabb) );
    cudaEventRecord(start);
    get_collision_pairs<<<grid, block, nBoxesPerThread*2*(BLOCK_PADDED)*sizeof(Aabb)>>>(d_boxes, d_count, d_overlaps, N, guess, nBoxesPerThread, d_queries);
    // get_collision_pairs_old<<<grid, block>>>(d_boxes, d_count, d_overlaps, N, guess);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // cudaDeviceSynchronize();

    long long queries;
    cudaMemcpy(&queries, d_queries, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("queries: %llu\n", queries);
    printf("needed queries: %llu\n", (long long)N*(N-1)/2 );

    // int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    

    if (count > guess) //we went over
    {
        printf("Running again\n");
        cudaFree(d_overlaps);
        cudaMalloc((void**)&d_overlaps, sizeof(int2)*(count));
        reset_counter<<<1,1>>>(d_count);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        get_collision_pairs<<<grid, block, nBoxesPerThread*2*(BLOCK_PADDED)*sizeof(Aabb)>>>(d_boxes, d_count, d_overlaps, N, count, nBoxesPerThread, d_queries);
        // get_collision_pairs_old<<<grid, block>>>(d_boxes, d_count, d_overlaps, N, 2*count);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        // cudaDeviceSynchronize();
    }

    printf("Elapsed time: %.6f ms\n", milliseconds);
    printf("Collisions: %i\n", count);
    printf("Elapsed time: %.9f ms/collision\n", milliseconds/count);
    printf("Boxes: %i\n", N);
    printf("Elapsed time: %.9f ms/box\n", milliseconds/N);
    // printf("Elapsed time: %.15f us/query\n", (milliseconds*1000)/((long long)N*N/2));

    int2 * overlaps =  (int2*)malloc(sizeof(int2) * (count));
    gpuErrchk(cudaMemcpy( overlaps, d_overlaps, sizeof(int2)*(count), cudaMemcpyDeviceToHost));


    cudaFree(d_overlaps);
    // for (size_t i=0; i< count; i++)
    // {
    //     // finOverlaps.push_back(overlaps[i].x, overlaps[i].y);
    //     // finOverlaps.push_back(overlaps[i].y);
        
    //     const Aabb& a = boxes[overlaps[i].x];
    //     const Aabb& b = boxes[overlaps[i].y];
    //     if (a.type == Simplex::VERTEX && b.type == Simplex::FACE)
    //     {
    //         finOverlaps.push_back(a.ref_id);
    //         finOverlaps.push_back(b.ref_id);
    //     }
    //     else if (a.type == Simplex::FACE && b.type == Simplex::VERTEX)
    //     {
    //         finOverlaps.push_back(b.ref_id);
    //         finOverlaps.push_back(a.ref_id);
    //     }
    //     else if (a.type == Simplex::EDGE && b.type == Simplex::EDGE)
    //     {   
    //         finOverlaps.push_back(min(a.ref_id, b.ref_id));
    //         finOverlaps.push_back(max(a.ref_id, b.ref_id));
    //     }
    // }

    printf("Total(filt.) overlaps: %lu\n", finOverlaps.size() / 2);
    free(overlaps);
    // free(counter);
    // free(counter);
    cudaFree(d_count);
    cudaDeviceReset();

}
//  // // //////// / / // / / // / // // / //  /

struct sort_sweepmarker_x
{
  __host__ __device__
  bool operator()(const SweepMarker &a, const SweepMarker &b) const {
    return (a.x < b.x);}
};

struct sort_aabb_x
{
  __host__ __device__
  bool operator()(const Aabb &a, const Aabb &b) const {
    return (a.min.x < b.min.x);}
};

// DEPRECATED
// void run_sweep(const Aabb* boxes, int N, int nbox, vector<pair<int,int>>& finOverlaps, int& threads)
// {
//     int devId = 0;
//     cudaSetDevice(devId);

//     int smemSize;

//     setup(devId, smemSize, threads, nbox);

//     finOverlaps.clear();
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop); 

//     // Allocate boxes to GPU 
//     Aabb * d_boxes;
//     cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
//     cudaMemcpy(d_boxes, boxes, sizeof(Aabb)*N, cudaMemcpyHostToDevice);

//     // Allocate counter to GPU + set to 0 collisions
//     int * d_count;
//     cudaMalloc((void**)&d_count, sizeof(int));
//     reset_counter<<<1,1>>>(d_count);
//     cudaDeviceSynchronize();

//     dim3 block(threads);
//     int grid_dim_1d = (N / threads + 1); 
//     dim3 grid( grid_dim_1d );
//     printf("Grid dim (1D): %i\n", grid_dim_1d);
//     printf("Box size: %i\n", sizeof(Aabb));
//     printf("SweepMarker size: %i\n", sizeof(SweepMarker));

//     // int* d_index;
//     // cudaMalloc((void**)&d_index, sizeof(int)*(N));
//     int* rank;
//     cudaMalloc((void**)&rank, sizeof(int)*(1*N));

//     int* rank_x = &rank[0];
//     // int* rank_y = &rank[N];
//     // int* rank_z = &rank[2*N];

//     // Translate boxes -> SweepMarkers
//     cudaEventRecord(start);
//     build_index<<<grid,block>>>(d_boxes, N, rank_x);
//     // build_index<<<grid,block>>>(d_boxes, N, rank_y);
//     // build_index<<<grid,block>>>(d_boxes, N, rank_z);
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);

//     printf("Elapsed time for build: %.6f ms\n", milliseconds);

//     // Thrust sort (can be improved by sort_by_key)
//     cudaEventRecord(start);
//     // thrust::sort(thrust::device, d_axis, d_axis + N, sort_sweepmarker_x() );
//     try{
//         thrust::sort_by_key(thrust::device, d_boxes, d_boxes + N, rank_x, sort_aabb_x() );
//         }
//     catch (thrust::system_error &e){
//         printf("Error: %s \n",e.what());}
    
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);

//     printf("Elapsed time for sort: %.6f ms\n", milliseconds);

//     // Test print some sorted output
//     // print_sort_axis<<<1,1>>>(d_boxes,rank_x, 5);
//     cudaDeviceSynchronize();

//     // Find overlapping pairs
//     int guess = 0;
//     int2 * d_overlaps;
//     cudaMalloc((void**)&d_overlaps, sizeof(int2)*(guess));

//     int count;
//     retrieve_collision_pairs<<<grid, block, smemSize>>>(d_boxes, rank_x, d_count, d_overlaps, N, guess, nbox);
//     cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();

//     if (count > guess) //we went over
//     {
//         printf("Running again\n");
//         cudaFree(d_overlaps);
//         cudaMalloc((void**)&d_overlaps, sizeof(int2)*(count));
//         reset_counter<<<1,1>>>(d_count);
//         cudaDeviceSynchronize();
//         cudaEventRecord(start);
//         retrieve_collision_pairs<<<grid, block, smemSize>>>(d_boxes, rank_x, d_count, d_overlaps, N, count, nbox);
//         cudaEventRecord(stop);
//         cudaEventSynchronize(stop);
//         print_overlap_start<<<1,1>>>(d_overlaps); 
//         cudaDeviceSynchronize();
//         milliseconds = 0;
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         printf("Elapsed time for findoverlaps: %.6f ms\n", milliseconds);
//     }
//     // int count;
//     cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();

//     printf("Elapsed time: %.6f ms\n", milliseconds);
//     printf("Collisions: %i\n", count);
//     printf("Elapsed time: %.9f ms/collision\n", milliseconds/count);
//     printf("Boxes: %i\n", N);
//     printf("Elapsed time: %.9f ms/box\n", milliseconds/N);

//     int2 * overlaps =  (int2*)malloc(sizeof(int2) * (count));
//     cudaMemcpy( overlaps, d_overlaps, sizeof(int2)*(count), cudaMemcpyDeviceToHost);

//     printf("Final count: %i\n", count);

//     cudaFree(d_overlaps);
//     for (size_t i=0; i < count; i++)
//     {
//         finOverlaps.emplace_back(overlaps[i].x, overlaps[i].y);
//         // finOverlaps.push_back(overlaps[i].y);
        
//         // const Aabb& a = boxes[overlaps[i].x];
//         // const Aabb& b = boxes[overlaps[i].y];
//         // if (a.type == Simplex::VERTEX && b.type == Simplex::FACE)
//         // {
//         //     finOverlaps.emplace_back(a.ref_id, b.ref_id);
//         // }
//         // else if (a.type == Simplex::FACE && b.type == Simplex::VERTEX)
//         // {
//         //     finOverlaps.emplace_back(b.ref_id, a.ref_id);
//         // }
//         // else if (a.type == Simplex::EDGE && b.type == Simplex::EDGE)
//         // {
//         //     finOverlaps.emplace_back(min(a.ref_id, b.ref_id), max(a.ref_id, b.ref_id));
//         // }
//     }

//     printf("Total(filt.) overlaps: %lu\n", finOverlaps.size() );
//     free(overlaps);
//     // free(counter);
//     // free(counter);
//     cudaFree(d_overlaps);
//     cudaFree(d_count); 

//     cudaDeviceReset();
// }

// MULTI GPU SWEEP SUPPORT
void merge_local_overlaps(
    const tbb::enumerable_thread_specific<tbb::concurrent_vector<std::pair<int,int>>>& storages,
    std::vector<std::pair<int,int>>& overlaps)
{
    overlaps.clear();
    size_t num_overlaps = overlaps.size();
    for (const auto& local_overlaps : storages) {
        num_overlaps += local_overlaps.size();
    }
    // serial merge!
    overlaps.reserve(num_overlaps);
    for (const auto& local_overlaps : storages) {
        overlaps.insert(
            overlaps.end(), local_overlaps.begin(), local_overlaps.end());
    }
}

void run_sweep_multigpu(const Aabb* boxes, int N, int nbox, vector<pair<int, int>>& finOverlaps, int& threads, int & devcount)
{
    cout<<"default threads "<<tbb::task_scheduler_init::default_num_threads()<<endl;
    tbb::enumerable_thread_specific<tbb::concurrent_vector<pair<int,int>>> storages;

    float milliseconds = 0;
    int device_init_id = 0;

    int smemSize;
    setup(device_init_id, smemSize, threads, nbox);

    cudaSetDevice(device_init_id);

    finOverlaps.clear();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    // Allocate boxes to GPU 
    Aabb * d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb)*N, cudaMemcpyHostToDevice);

    dim3 block(threads);
    int grid_dim_1d = (N / threads + 1); 
    dim3 grid( grid_dim_1d );
    printf("Grid dim (1D): %i\n", grid_dim_1d);
    printf("Box size: %i\n", sizeof(Aabb));

    // int* rank;
    // cudaMalloc((void**)&rank, sizeof(int)*(1*N));

    // int* rank_x = &rank[0];
    // int* rank_y = &rank[N];
    // int* rank_z = &rank[2*N];

    // Translate boxes -> SweepMarkers

    // cudaEventRecord(start);
    // build_index<<<grid,block>>>(d_boxes, N, rank_x);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    
    // cudaEventElapsedTime(&milliseconds, start, stop);

    // printf("Elapsed time for build: %.6f ms\n", milliseconds);

    // Thrust sort (can be improved by sort_by_key)
    cudaEventRecord(start);
    try{
        // thrust::sort_by_key(thrust::device, d_boxes, d_boxes + N, rank_x, sort_aabb_x() );
        thrust::sort(thrust::device, d_boxes, d_boxes + N, sort_aabb_x() );
        }
    catch (thrust::system_error &e){
        printf("Error: %s \n",e.what());}
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time for sort: %.6f ms\n", milliseconds);


    // Test print some sorted output
    // print_sort_axis<<<1,1>>>(d_boxes, 5);
    cudaDeviceSynchronize();
    

    int devices_count;
    cudaGetDeviceCount(&devices_count);
    // devices_count-=2;
    devices_count = devcount ? devcount : devices_count;
    int range = ceil( (float)N / devices_count); 

    // free(start);
    // free(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEvent_t starts[devices_count];
    cudaEvent_t stops[devices_count];
    float millisecondss[devices_count];

    tbb::parallel_for(0, devices_count, 1, [&](int & device_id)    {

        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        printf("%s -> unifiedAddressing = %d\n", prop.name, prop.unifiedAddressing);

        cudaSetDevice(device_id);

        // cudaEvent_t start, stop;
        cudaEventCreate(&starts[device_id]);
        cudaEventCreate(&stops[device_id]); 

        int is_able;

        cudaDeviceCanAccessPeer(&is_able, device_id, device_init_id);
        cudaDeviceSynchronize();
        if (is_able)
        { 
            cudaDeviceEnablePeerAccess(device_init_id, 0); 
            cudaDeviceSynchronize(); 
        }
        else if (device_init_id != device_id)
            printf("Device %i cant access Device %i\n", device_id, device_init_id);

        int range_start  = range*device_id;
        int range_end = range*(device_id + 1);
        printf("device_id: %i [%i, %i)\n", device_id, range_start, range_end);
        

        Aabb * d_b;
        cudaMalloc((void**)&d_b, sizeof(Aabb)*N);
        cudaMemcpy(d_b, d_boxes, sizeof(Aabb)*N, cudaMemcpyDefault);
        cudaDeviceSynchronize();  
       
        cudaDeviceCanAccessPeer(&is_able, device_id, device_init_id);
        cudaDeviceSynchronize();
        if (is_able)
        { 
            cudaDeviceDisablePeerAccess(device_init_id); 
            cudaDeviceSynchronize();  
        }
        else if (device_init_id != device_id)
            printf("Device %i cant access Device %i\n", device_id, device_init_id);

        
        // Allocate counter to GPU + set to 0 collisions
        int * d_count;
        gpuErrchk(cudaMalloc((void**)&d_count, sizeof(int)));
        gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));
        gpuErrchk( cudaGetLastError() );   

        // Find overlapping pairs
        int guess = N*360;
        printf("Guess %i\n", guess);

        int2 * d_overlaps;
        cudaMalloc((void**)&d_overlaps, sizeof(int2)*(guess));
        gpuErrchk( cudaGetLastError() ); 
        
        int grid_dim_1d = (range / threads + 1); 
        dim3 grid( grid_dim_1d );

        int count;
        cudaEventRecord(starts[device_id]);
        retrieve_collision_pairs<<<grid, block, smemSize>>>(d_b, d_count, d_overlaps, N, guess, nbox, range_start, range_end);
        cudaEventRecord(stops[device_id]);
        cudaEventSynchronize(stops[device_id]);
        cudaEventElapsedTime(&millisecondss[device_id], starts[device_id], stops[device_id]);
        cudaDeviceSynchronize();
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        printf("count for device %i : %i\n", device_id, count);
    
        if (count > guess){
            printf("Running again\n");
            cudaFree(d_overlaps);
            cudaMalloc((void**)&d_overlaps, sizeof(int2)*(count));
            // cudaMemset(d_overlaps, 0, sizeof(int2)*(count));
            cudaMemset(d_count, 0, sizeof(int));
            cudaEventRecord(starts[device_id]);
            retrieve_collision_pairs<<<grid, block, smemSize>>>(d_b, d_count, d_overlaps, N, count, nbox, range_start, range_end);
            cudaEventRecord(stops[device_id]);
            cudaEventSynchronize(stops[device_id]);
            cudaEventElapsedTime(&millisecondss[device_id], starts[device_id], stops[device_id]);
            cudaDeviceSynchronize();
            cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
            printf("count2 for device %i : %i\n", device_id, count);
        }

        
        // printf("Elapsed time: %.9f ms/collision\n", milliseconds/count);
        // printf("Boxes: %i\n", N);
        // printf("Elapsed time: %.9f ms/box\n", milliseconds/N);

        // int2 * overlaps = new int2[count];
        int2* overlaps =  (int2*)malloc(sizeof(int2) * count);
        gpuErrchk(cudaMemcpy( overlaps, d_overlaps, sizeof(int2)*(count), cudaMemcpyDeviceToHost));
        gpuErrchk( cudaGetLastError() ); 

        printf("Final count for device %i:  %i\n", device_id, count);

        auto& local_overlaps = storages.local();
        // local_overlaps.reserve(local_overlaps.size() + count);
        
        auto is_face = [&](Aabb x){return x.vertexIds.z >= 0;};
        auto is_edge = [&](Aabb x){return x.vertexIds.z < 0 && x.vertexIds.y >= 0 ;};
        auto is_vertex = [&](Aabb x){return x.vertexIds.z < 0  && x.vertexIds.y < 0;};
        
        
        for (size_t i=0; i < count; i++)
        {
            // local_overlaps.emplace_back(overlaps[i].x, overlaps[i].y);
            // finOverlaps.push_back();
            
            Aabb a = boxes[overlaps[i].x];
            Aabb b = boxes[overlaps[i].y];
            
            if (is_vertex(a) && is_face(b)) //vertex, face
            {
                local_overlaps.emplace_back(a.ref_id, b.ref_id);
            }
            else if (is_face(a) && is_vertex(b))
            {
                local_overlaps.emplace_back(b.ref_id, a.ref_id);
            }
            else if (is_edge(a) && is_edge(b))
            {
                local_overlaps.emplace_back(min(a.ref_id, b.ref_id), max(a.ref_id, b.ref_id));
            }
        }
        
        printf("Total(filt.) overlaps for devid %i: %i\n", device_id, local_overlaps.size());
        // delete [] overlaps;
        // free(overlaps);
        
        // // free(counter);
        // // free(counter);
        // cudaFree(d_overlaps);
        // cudaFree(d_count); 
        // cudaFree(d_b);
        // cudaFree(d_r);
        // cudaDeviceReset();

    }); //end tbb for loop

    merge_local_overlaps(storages, finOverlaps);

    float longest = 0;
    for (int i=0; i<devices_count; i++)
    {
        for (int j=0; j<devices_count; j++)
        {
            cudaEventElapsedTime(&milliseconds, starts[i], stops[j]);
            longest = milliseconds > longest ? milliseconds : longest;
        }

    }
    printf("\n");
    printf("Elapsed time: %.6f ms\n", longest);
    printf("Merged overlaps: %i\n", finOverlaps.size());
    printf("\n");

}