#include <stq/gpu/simulation.cuh>

#include <stq/gpu/collision.cuh>
#include <stq/gpu/queue.cuh>
#include <stq/gpu/sweep.cuh>
#include <stq/gpu/timer.cuh>
#include <stq/gpu/memory.cuh>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

#include <spdlog/spdlog.h>

namespace stq::gpu {

extern MemHandler *memhandle;

void setup(int devId, int &smemSize, int &threads, int &nboxes);

void run_collision_counter(Aabb *boxes, int N) {

  // int N = 200000;
  // Aabb boxes[N];
  // for (int i = 0; i<N; i++)
  // {
  //     boxes[i] = Aabb(i);
  //     // spdlog::trace("box {:d} created", boxes[i].id);
  // }

  // Allocate boxes to GPU
  Aabb *d_boxes;
  cudaMalloc((void **)&d_boxes, sizeof(Aabb) * N);
  cudaMemcpy(d_boxes, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);

  // Allocate counter to GPU + set to 0 collisions
  int *d_counter;
  cudaMalloc((void **)&d_counter, sizeof(int));
  reset_counter<<<1, 1>>>(d_counter);
  cudaDeviceSynchronize();

  int collisions;
  // cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

  // int bytes_mem_intrfce = 352 >> 3;
  // int mem_clock_rate = 1376 << 1;
  // float bandwidth_mem_theor = (mem_clock_rate * bytes_mem_intrfce) / pow(10,
  // 3);

  // Set up timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Get number of collisions
  cudaEventRecord(start);
  count_collisions<<<1, 1>>>(d_boxes, d_counter, N);
  cudaEventRecord(stop);
  cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  spdlog::trace("(count_collisions<<<1,1>>>)\n");
  spdlog::trace("Elapsed time: {:.6f} ms", milliseconds);
  spdlog::trace("Elapsed time: {:.6f} ms/c", milliseconds / collisions);
  spdlog::trace("Collision: {:d}", collisions);
  spdlog::trace("Effective Bandwidth (GB/s): {:.6f} (GB/s)",
                32 * 2 / milliseconds / 1e6);

  reset_counter<<<1, 1>>>(d_counter);
  cudaEventRecord(start);
  count_collisions<<<1, 1024>>>(d_boxes, d_counter, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  spdlog::trace("(count_collisions<<<1,1024>>>)");
  spdlog::trace("Elapsed time: {:.6f} ms", milliseconds);
  spdlog::trace("Elapsed time: {:.6f} ms/c", milliseconds / collisions);
  spdlog::trace("Collision: {:d}", collisions);

  reset_counter<<<1, 1>>>(d_counter);
  cudaEventRecord(start);
  count_collisions<<<2, 1024>>>(d_boxes, d_counter, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  spdlog::trace("(count_collisions<<<2,1024>>>)");
  spdlog::trace("Elapsed time: {:.6f} ms", milliseconds);
  spdlog::trace("Elapsed time: {:.6f} ms/c", milliseconds / collisions);
  spdlog::trace("Collision: {:d}", collisions);

  reset_counter<<<1, 1>>>(d_counter);
  cudaEventRecord(start);
  count_collisions<<<56, 1024>>>(d_boxes, d_counter, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  spdlog::trace("(count_collisions<<<56,1024>>>)");
  spdlog::trace("Elapsed time: {:.6f} ms", milliseconds);
  spdlog::trace("Elapsed time: {:.9f} ms/c", milliseconds / collisions);
  spdlog::trace("Collision: {:d}", collisions);

  reset_counter<<<1, 1>>>(d_counter);
  cudaEventRecord(start);
  count_collisions<<<256, 1024>>>(d_boxes, d_counter, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaMemcpy(&collisions, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  spdlog::trace("(count_collisions<<<256,1024>>>)");
  spdlog::trace("Elapsed time: {:.6f} ms", milliseconds);
  spdlog::trace("Elapsed time: {:.9f} ms/c", milliseconds / collisions);
  spdlog::trace("Collision: {:d}", collisions);
  return;
  // spdlog::trace("%zu", sizeof(Aabb));

  // Retrieve count from GPU and print out
  // int counter;
  // cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  // spdlog::trace("count: {:d}", counter);
  // return 0;
}

void run_scaling(const Aabb *boxes, int N, int desiredBoxesPerThread,
                 std::vector<unsigned long> &finOverlaps) {

  int devId = 0;
  cudaSetDevice(devId);

  int smemSize;
  int threads;

  setup(devId, smemSize, threads, desiredBoxesPerThread);
  const int nBoxesPerThread =
    desiredBoxesPerThread ? desiredBoxesPerThread
                          : smemSize / sizeof(Aabb) / (2 * (BLOCK_PADDED));
  spdlog::trace("Boxes per Thread: {:d}", nBoxesPerThread);

  finOverlaps.clear();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // guess overlaps size
  int guess = 0;

  // Allocate boxes to GPU
  Aabb *d_boxes;
  cudaMalloc((void **)&d_boxes, sizeof(Aabb) * N);
  cudaMemcpy(d_boxes, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);

  // Allocate counter to GPU + set to 0 collisions
  int *d_count;
  cudaMalloc((void **)&d_count, sizeof(int));
  reset_counter<<<1, 1>>>(d_count);
  cudaDeviceSynchronize();

  // Count collisions
  count_collisions<<<1, 1>>>(d_boxes, d_count, N);
  int count;
  cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  reset_counter<<<1, 1>>>(d_count);
  spdlog::trace("Total collisions from counting: {:d}", count);

  int2 *d_overlaps;
  cudaMalloc((void **)&d_overlaps, sizeof(int2) * (guess));

  dim3 block(BLOCK_SIZE_1D, BLOCK_SIZE_1D);
  // dim3 grid ( (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D,
  // (N+BLOCK_SIZE_1D)/BLOCK_SIZE_1D );
  int grid_dim_1d = (N + BLOCK_SIZE_1D) / BLOCK_SIZE_1D / nBoxesPerThread;
  dim3 grid(grid_dim_1d, grid_dim_1d);
  spdlog::trace("Grid dim (1D): {:d}", grid_dim_1d);
  spdlog::trace("Box size: {:d}", sizeof(Aabb));

  long long *d_queries;
  cudaMalloc((void **)&d_queries, sizeof(long long) * (1));
  reset_counter<<<1, 1>>>(d_queries);

  spdlog::trace("Shared mem alloc: {:d} B",
                nBoxesPerThread * 2 * (BLOCK_PADDED) * sizeof(Aabb));
  cudaEventRecord(start);
  get_collision_pairs<<<grid, block,
                        nBoxesPerThread * 2 * (BLOCK_PADDED) * sizeof(Aabb)>>>(
    d_boxes, d_count, d_overlaps, N, guess, nBoxesPerThread, d_queries);
  // get_collision_pairs_old<<<grid, block>>>(d_boxes, d_count, d_overlaps, N,
  // guess);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // cudaDeviceSynchronize();

  long long queries;
  cudaMemcpy(&queries, d_queries, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  spdlog::trace("queries: {:d}", queries);
  spdlog::trace("needed queries: {:d}", (long long)N * (N - 1) / 2);

  // int count;
  cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (count > guess) // we went over
  {
    spdlog::trace("Running again\n");
    cudaFree(d_overlaps);
    cudaMalloc((void **)&d_overlaps, sizeof(int2) * (count));
    reset_counter<<<1, 1>>>(d_count);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    get_collision_pairs<<<
      grid, block, nBoxesPerThread * 2 * (BLOCK_PADDED) * sizeof(Aabb)>>>(
      d_boxes, d_count, d_overlaps, N, count, nBoxesPerThread, d_queries);
    // get_collision_pairs_old<<<grid, block>>>(d_boxes, d_count, d_overlaps, N,
    // 2*count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // cudaDeviceSynchronize();
  }

  spdlog::trace("Elapsed time: {:.6f} ms", milliseconds);
  spdlog::trace("Collisions: {:d}", count);
  spdlog::trace("Elapsed time: {:.9f} ms/collision", milliseconds / count);
  spdlog::trace("Boxes: {:d}", N);
  spdlog::trace("Elapsed time: {:.9f} ms/box", milliseconds / N);
  // spdlog::trace("Elapsed time: {:.15f} us/query", (milliseconds*1000)/((long
  // long)N*N/2));

  int2 *overlaps = (int2 *)malloc(sizeof(int2) * (count));
  gpuErrchk(cudaMemcpy(overlaps, d_overlaps, sizeof(int2) * (count),
                       cudaMemcpyDeviceToHost));

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

  spdlog::trace("Total(filt.) overlaps: {:d}", finOverlaps.size() / 2);
  free(overlaps);
  // free(counter);
  // free(counter);
  cudaFree(d_count);
  cudaDeviceReset();
}

struct sorter {};

struct sort_aabb_x : sorter {
  __device__ bool operator()(const Aabb &a, const Aabb &b) const {
    return (a.min.x < b.min.x);
  }

  __device__ bool operator()(const Scalar3 &a, const Scalar3 &b) const {
    return (a.x < b.x);
  }

  __device__ bool operator()(const Scalar2 &a, const Scalar2 &b) const {
    return (a.x < b.x);
  }

  __device__ bool operator()(const RankBox &a, const RankBox &b) const {
    return (a.aabb->min.x < b.aabb->min.x);
  }
};

typedef tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>>
  ThreadSpecificOverlaps;

void merge_local_overlaps(const ThreadSpecificOverlaps &storages,
                          std::vector<std::pair<int, int>> &overlaps) {
  overlaps.clear();
  size_t num_overlaps = overlaps.size();
  for (const auto &local_overlaps : storages) {
    num_overlaps += local_overlaps.size();
  }
  // serial merge!
  overlaps.reserve(num_overlaps);
  for (const auto &local_overlaps : storages) {
    overlaps.insert(overlaps.end(), local_overlaps.begin(),
                    local_overlaps.end());
  }
}

void run_sweep_multigpu(const Aabb *boxes, int N, int nbox,
                        std::vector<std::pair<int, int>> &finOverlaps,
                        int &threads, int &devcount) {
  spdlog::critical("default threads {}", tbb::info::default_concurrency());
  ThreadSpecificOverlaps storages;

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
  Aabb *d_boxes;
  cudaMalloc((void **)&d_boxes, sizeof(Aabb) * N);
  cudaMemcpy(d_boxes, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);

  dim3 block(threads);
  int grid_dim_1d = (N / threads + 1);
  dim3 grid(grid_dim_1d);
  spdlog::trace("Grid dim (1D): {:d}", grid_dim_1d);
  spdlog::trace("Box size: {:d}", sizeof(Aabb));

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

  // spdlog::trace("Elapsed time for build: {:.6f} ms", milliseconds);

  // Thrust sort (can be improved by sort_by_key)
  cudaEventRecord(start);
  try {
    // thrust::sort_by_key(thrust::device, d_boxes, d_boxes + N, rank_x,
    // sort_aabb_x() );
    thrust::sort(thrust::device, d_boxes, d_boxes + N, sort_aabb_x());
  } catch (thrust::system_error &e) {
    spdlog::trace("Error: {:s} ", e.what());
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  spdlog::trace("Elapsed time for sort: {:.6f} ms", milliseconds);

  // Test print some sorted output
  // print_sort_axis<<<1,1>>>(d_boxes, 5);
  cudaDeviceSynchronize();

  int devices_count;
  cudaGetDeviceCount(&devices_count);
  // devices_count-=2;
  devices_count = devcount ? devcount : devices_count;
  int range = ceil((float)N / devices_count);

  // free(start);
  // free(stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaEvent_t starts[devices_count];
  cudaEvent_t stops[devices_count];
  float millisecondss[devices_count];

  tbb::parallel_for(0, devices_count, 1, [&](int &device_id) {
    auto &local_overlaps = storages.local();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    spdlog::trace("{:s} -> unifiedAddressing = {:d}", prop.name,
                  prop.unifiedAddressing);

    cudaSetDevice(device_id);

    // cudaEvent_t start, stop;
    cudaEventCreate(&starts[device_id]);
    cudaEventCreate(&stops[device_id]);

    int is_able;

    cudaDeviceCanAccessPeer(&is_able, device_id, device_init_id);
    cudaDeviceSynchronize();
    if (is_able) {
      cudaDeviceEnablePeerAccess(device_init_id, 0);
      cudaDeviceSynchronize();
    } else if (device_init_id != device_id)
      spdlog::trace("Device {:d} cant access Device {:d}", device_id,
                    device_init_id);

    int range_start = range * device_id;
    int range_end = range * (device_id + 1);
    spdlog::trace("device_id: {:d} [{:d}, {:d})", device_id, range_start,
                  range_end);

    Aabb *d_b;
    cudaMalloc((void **)&d_b, sizeof(Aabb) * N);
    cudaMemcpy(d_b, d_boxes, sizeof(Aabb) * N, cudaMemcpyDefault);
    cudaDeviceSynchronize();

    cudaDeviceCanAccessPeer(&is_able, device_id, device_init_id);
    cudaDeviceSynchronize();
    if (is_able) {
      cudaDeviceDisablePeerAccess(device_init_id);
      cudaDeviceSynchronize();
    } else if (device_init_id != device_id)
      spdlog::trace("Device {:d} cant access Device {:d}", device_id,
                    device_init_id);

    // Allocate counter to GPU + set to 0 collisions
    int *d_count;
    gpuErrchk(cudaMalloc((void **)&d_count, sizeof(int)));
    gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));
    gpuErrchk(cudaGetLastError());

    // Find overlapping pairs
    int guess = N * 200;
    spdlog::trace("Guess {:d}", guess);

    int2 *d_overlaps;
    cudaMalloc((void **)&d_overlaps, sizeof(int2) * (guess));
    gpuErrchk(cudaGetLastError());

    int grid_dim_1d = (range / threads + 1);
    dim3 grid(grid_dim_1d);

    int count;
    cudaEventRecord(starts[device_id]);
    retrieve_collision_pairs<<<grid, block, smemSize>>>(
      d_b, d_count, d_overlaps, N, guess, nbox, range_start, range_end);
    cudaEventRecord(stops[device_id]);
    cudaEventSynchronize(stops[device_id]);
    cudaEventElapsedTime(&millisecondss[device_id], starts[device_id],
                         stops[device_id]);
    cudaDeviceSynchronize();
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    spdlog::trace("count for device {:d} : {:d}", device_id, count);

    if (count > guess) {
      spdlog::trace("Running again");
      cudaFree(d_overlaps);
      cudaMalloc((void **)&d_overlaps, sizeof(int2) * (count));
      // cudaMemset(d_overlaps, 0, sizeof(int2)*(count));
      cudaMemset(d_count, 0, sizeof(int));
      cudaEventRecord(starts[device_id]);
      retrieve_collision_pairs<<<grid, block, smemSize>>>(
        d_b, d_count, d_overlaps, N, count, nbox, range_start, range_end);
      cudaEventRecord(stops[device_id]);
      cudaEventSynchronize(stops[device_id]);
      cudaEventElapsedTime(&millisecondss[device_id], starts[device_id],
                           stops[device_id]);
      cudaDeviceSynchronize();
      cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
      spdlog::trace("count2 for device {:d} : {:d}", device_id, count);
    }

    // spdlog::trace("Elapsed time: {:.9f} ms/collision", milliseconds/count);
    // spdlog::trace("Boxes: {:d}", N);
    // spdlog::trace("Elapsed time: {:.9f} ms/box", milliseconds/N);

    // int2 * overlaps = new int2[count];
    int2 *overlaps = (int2 *)malloc(sizeof(int2) * count);
    gpuErrchk(cudaMemcpy(overlaps, d_overlaps, sizeof(int2) * (count),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaGetLastError());

    spdlog::trace("Final count for device {:d}:  {:d}", device_id, count);

    // local_overlaps.reserve(local_overlaps.size() + count);

    // auto is_face = [&](Aabb x){return x.vertexIds.z >= 0;};
    // auto is_edge = [&](Aabb x){return x.vertexIds.z < 0 && x.vertexIds.y >= 0
    // ;}; auto is_vertex = [&](Aabb x){return x.vertexIds.z < 0  &&
    // x.vertexIds.y < 0;};

    for (size_t i = 0; i < count; i++) {
      // local_overlaps.emplace_back(overlaps[i].x, overlaps[i].y);
      // finOverlaps.push_back();
      int aid = overlaps[i].x;
      int bid = overlaps[i].y;
      Aabb a = boxes[aid];
      Aabb b = boxes[bid];

      if (is_vertex(a) && is_face(b)) // vertex, face
        local_overlaps.emplace_back(aid, bid);
      else if (is_edge(a) && is_edge(b))
        local_overlaps.emplace_back(min(aid, bid), max(aid, bid));
      else if (is_face(a) && is_vertex(b))
        local_overlaps.emplace_back(bid, aid);
    }

    spdlog::trace("Total(filt.) overlaps for devid {:d}: {:d}", device_id,
                  local_overlaps.size());
    // delete [] overlaps;
    // free(overlaps);

    // // free(counter);
    // // free(counter);
    // cudaFree(d_overlaps);
    // cudaFree(d_count);
    // cudaFree(d_b);
    // cudaFree(d_r);
    // cudaDeviceReset();
  }); // end tbb for loop

  merge_local_overlaps(storages, finOverlaps);

  float longest = 0;
  for (int i = 0; i < devices_count; i++) {
    for (int j = 0; j < devices_count; j++) {
      cudaEventElapsedTime(&milliseconds, starts[i], stops[j]);
      longest = milliseconds > longest ? milliseconds : longest;
    }
  }
  printf("\n");
  spdlog::trace("Elapsed time: {:.6f} ms", longest);
  spdlog::trace("Merged overlaps: {:d}", finOverlaps.size());
  printf("\n");
}

void run_sweep_sharedqueue(const Aabb *boxes, MemHandler *memhandle, int N,
                           int nbox,
                           std::vector<std::pair<int, int>> &finOverlaps,
                           int2 *&d_overlaps, int *&d_count, int &threads,
                           int &tidstart, int &devcount, const int memlimit) {
  cudaDeviceSynchronize();
  spdlog::trace("Number of boxes: {:d}", N);

  if (!memhandle->MAX_OVERLAP_CUTOFF)
    memhandle->MAX_OVERLAP_CUTOFF = N;
  if (memlimit) {
    memhandle->limitGB = memlimit;
    spdlog::trace("Limit set to {:d}", memhandle->limitGB);
  }

  int device_init_id = 0;

  int smemSize;
  setup(device_init_id, smemSize, threads, nbox);

  cudaSetDevice(device_init_id);

  // Allocate boxes to GPU
  Aabb *d_boxes;
  cudaMalloc((void **)&d_boxes, sizeof(Aabb) * N);
  cudaMemcpy(d_boxes, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);

  int grid_dim_1d = (N / threads + 1);
  spdlog::trace("Grid dim (1D): {:d}", grid_dim_1d);
  spdlog::trace("Box size: {:d}", sizeof(Aabb));
  spdlog::trace("Scalar3 size: {:d}", sizeof(Scalar3));
  spdlog::trace("sizeof(queue) size: {:d}", sizeof(Queue));

  Scalar2 *d_sm;
  cudaMalloc((void **)&d_sm, sizeof(Scalar2) * N);

  MiniBox *d_mini;
  cudaMalloc((void **)&d_mini, sizeof(MiniBox) * N);

  // mean of all box points (used to find best axis)
  //   Scalar3 *d_mean;
  //   cudaMalloc((void **)&d_mean, sizeof(Scalar3));
  //   cudaMemset(d_mean, 0, sizeof(Scalar3));

  //   // recordLaunch("create_ds", grid_dim_1d, threads, smemSize, create_ds,
  //   // d_boxes, d_sm, d_mini, N, d_mean);
  //   recordLaunch("calc_mean", grid_dim_1d, threads, smemSize, calc_mean,
  //   d_boxes,
  //                d_mean, N);

  //   // temporary
  //   Scalar3 mean;
  //   cudaMemcpy(&mean, d_mean, sizeof(Scalar3),
  //   cudaMemcpyDeviceToHost); spdlog::trace("mean: x {:.6f} y {:.6f} z
  //   {:.6f}", mean.x, mean.y, mean.z);

  //   // calculate variance and determine which axis to sort on
  //   Scalar3 *d_var; // 2 vertices per box
  //   cudaMalloc((void **)&d_var, sizeof(Scalar3));
  //   cudaMemset(d_var, 0, sizeof(Scalar3));
  //   // calc_variance(boxes, d_var, N, d_mean);
  //   recordLaunch("calc_variance", grid_dim_1d, threads, smemSize,
  //   calc_variance,
  //                d_boxes, d_var, N, d_mean);
  //   cudaDeviceSynchronize();

  //   Scalar3 var3d;
  //   cudaMemcpy(&var3d, d_var, sizeof(Scalar3),
  //   cudaMemcpyDeviceToHost); float maxVar = max(max(var3d.x, var3d.y),
  //   var3d.z);

  //   spdlog::trace("var: x {:.6f} y {:.6f} z {:.6f}", var3d.x, var3d.y,
  //   var3d.z);

  Dimension axis;
  //   if (maxVar == var3d.x)
  //     axis = x;
  //   else if (maxVar == var3d.y)
  //     axis = y;
  //   else
  //     axis = z;
  //   // hack
  axis = x;

  spdlog::trace("Axis: {:s}", axis == x ? "x" : (axis == y ? "y" : "z"));

  recordLaunch<Aabb *, Scalar2 *, MiniBox *, int, Dimension>(
    "create_ds", grid_dim_1d, threads, smemSize, create_ds, d_boxes, d_sm,
    d_mini, N, axis);

  try {
    thrust::sort_by_key(thrust::device, d_sm, d_sm + N, d_mini, sort_aabb_x());
  } catch (thrust::system_error &e) {
    spdlog::trace("Thrust error: {:s} ", e.what());
  }
  spdlog::trace("Thrust sort finished");

  gpuErrchk(cudaGetLastError());

  // MemHandler memhandle;
  // Guessing global collision output size
  // int guess = memhandle->MAX_OVERLAP_CUTOFF; // 200 * N;
  spdlog::trace("Guess cutoff: {:d}", memhandle->MAX_OVERLAP_CUTOFF);
  size_t overlaps_size = memhandle->MAX_OVERLAP_SIZE * sizeof(int2);
  spdlog::trace("overlaps_size: {:d}", overlaps_size);
  gpuErrchk(cudaGetLastError());

  int *d_start;
  int *d_end;
  // int boxes_done = tidstart;

  gpuErrchk(cudaMalloc((void **)&d_start, sizeof(int)));
  gpuErrchk(cudaMalloc((void **)&d_end, sizeof(int)));
  gpuErrchk(
    cudaMemcpy(d_start, &tidstart, sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(d_end, 0, sizeof(int)));
  gpuErrchk(cudaGetLastError());

  // int * d_count;
  gpuErrchk(cudaMalloc((void **)&d_count, sizeof(int)));
  gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));

  // Device memhandler to keep track of vars
  MemHandler *d_memhandle;
  gpuErrchk(cudaMalloc((void **)&d_memhandle, sizeof(MemHandler)));
  cudaMemcpy(d_memhandle, memhandle, sizeof(MemHandler),
             cudaMemcpyHostToDevice);

  // int2 * d_overlaps;
  spdlog::trace("Allocating overlaps memory");
  gpuErrchk(cudaMalloc((void **)&d_overlaps, overlaps_size));

  spdlog::trace("Starting two stage_queue");
  spdlog::trace("Starting tid {:d}", tidstart);
  recordLaunch<Scalar2 *, const MiniBox *, int2 *, int, int *, int *, int *,
               MemHandler *>("twostage_queue_1st", grid_dim_1d, threads,
                             twostage_queue, d_sm, d_mini, d_overlaps, N,
                             d_count, d_start, d_end, d_memhandle);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaGetLastError());

  int count;
  gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  spdlog::debug("1st count for device {:d}:  {:d}", device_init_id, count);

  int realcount;
  gpuErrchk(cudaMemcpy(&realcount, &(d_memhandle->realcount), sizeof(int),
                       cudaMemcpyDeviceToHost));
  spdlog::trace("Real count for device {:d}:  {:d}", device_init_id, realcount);

  // int diff = boxes_done;
  // gpuErrchk(
  //   cudaMemcpy(&boxes_done, d_end, sizeof(int), cudaMemcpyDeviceToHost));
  // diff = boxes_done - diff;

  spdlog::debug("realcount: {:d}, overlap_size {:d} -> Batching", realcount,
                memhandle->MAX_OVERLAP_SIZE);
  while (count > memhandle->MAX_OVERLAP_SIZE) {
    gpuErrchk(cudaFree(d_overlaps));

    memhandle->handleBroadPhaseOverflow(count);

    gpuErrchk(cudaMalloc((void **)&d_overlaps,
                         sizeof(int2) * (memhandle->MAX_OVERLAP_SIZE)));

    gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));
    gpuErrchk(cudaMemset(d_end, 0, sizeof(int)));

    cudaMemcpy(d_memhandle, memhandle, sizeof(MemHandler),
               cudaMemcpyHostToDevice);

    recordLaunch<Scalar2 *, const MiniBox *, int2 *, int, int *, int *, int *,
                 MemHandler *>("twostage_queue_1st", grid_dim_1d, threads,
                               twostage_queue, d_sm, d_mini, d_overlaps, N,
                               d_count, d_start, d_end, d_memhandle);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(&count, &(d_memhandle->realcount), sizeof(int),
    //                      cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&realcount, &(d_memhandle->realcount), sizeof(int),
                         cudaMemcpyDeviceToHost));
    spdlog::trace("Real count for loop:  {:d}", realcount);
    spdlog::trace("Count for loop:  {:d}", count);
    // gpuErrchk(
    //   cudaMemcpy(&boxes_done, d_end, sizeof(int), cudaMemcpyDeviceToHost));
    spdlog::debug("Count {:d}, max size {:d}", realcount,
                  memhandle->MAX_OVERLAP_SIZE);
  }
  // tidstart = boxes_done;
  tidstart += memhandle->MAX_OVERLAP_CUTOFF;
  // spdlog::trace("Next threadid start {:d}", tidstart);

  gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  // spdlog::trace("Final count for device {:d}:  {:d}", device_init_id, count);
  gpuErrchk(cudaMemcpy(d_count, &(d_memhandle->realcount), sizeof(int),
                       cudaMemcpyDeviceToDevice));
  // spdlog::trace("Final count for device {:d}:  {:d}", device_init_id, count);

  cudaFree(d_boxes);
  cudaFree(d_mini);
  cudaFree(d_sm);
  cudaFree(d_start);
  cudaFree(d_end);
  cudaFree(d_memhandle);

#ifdef KEEP_CPU_OVERLAPS
  int2 *overlaps = (int2 *)malloc(sizeof(int2) * count);
  gpuErrchk(cudaMemcpy(overlaps, d_overlaps, sizeof(int2) * (count),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaGetLastError());

  spdlog::trace("Final count for device {:d}:  {:d}", 0, count);

  finOverlaps.reserve(finOverlaps.size() + count);
  for (int i = 0; i < count; i++) {
    finOverlaps.emplace_back(overlaps[i].x, overlaps[i].y);
  }

  free(overlaps);

  spdlog::trace("Total(filt.) overlaps for devid {:d}: {:d}", 0,
                finOverlaps.size());
#endif
  spdlog::trace("Next threadstart {:d}", tidstart);
}
} // namespace stq::gpu