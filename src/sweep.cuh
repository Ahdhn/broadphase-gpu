#pragma once

//#include <bits/stdc++.h>

// need this to get tiled_partition > 32 threads
#define _CG_ABI_EXPERIMENTAL // enable experimental API

#include <cooperative_groups.h>

// #include <stq/gpu/aabb.cuh>
#include <stq/gpu/collision.cuh>
// #include <stq/gpu/memory.cuh>

namespace cg = cooperative_groups;

namespace stq::gpu {

__global__ void build_index(Aabb *boxes, int N, int *index);
__global__ void print_sort_axis(Aabb *axis, int C);
__global__ void retrieve_collision_pairs(const Aabb *const boxes, int *count,
                                         int2 *overlaps, int N, int guess,
                                         int nbox, int start = 0,
                                         int end = INT_MAX);
__global__ void print_overlap_start(int2 *overlaps);

// for balancing
__global__ void build_checker(Scalar3 *sortedmin, int2 *outpairs, int N,
                              int *count, int guess);
__global__ void create_ds(Aabb *boxes, Scalar2 *sortedmin, MiniBox *mini, int N,
                          Dimension axis);
__global__ void calc_variance(Aabb *boxes, Scalar3 *var, int N, Scalar3 *mean);
__global__ void calc_mean(Aabb *boxes, Scalar3 *mean, int N);
__global__ void twostage_queue(Scalar2 *sm, const MiniBox *const mini,
                               int2 *overlaps, int N, int *count, int *start,
                               int *end, MemHandler *memHandler);

// for pairing
__global__ void create_ds(Aabb *boxes, Scalar3 *sortedmin, MiniBox *mini, int N,
                          Scalar3 *mean);
__global__ void assign_rank_c(RankBox *rankboxes, int N);
__global__ void register_rank_y(RankBox *rankboxes, int N);
__global__ void register_rank_x(RankBox *rankboxes, int N);
__global__ void create_rankbox(Aabb *boxes, RankBox *rankboxes, int N);
__global__ void build_checker2(const RankBox *const rankboxes, int2 *out, int N,
                               int *count, int guess);
__global__ void print_stats(RankBox *rankboxes, int N);

} // namespace stq::gpu