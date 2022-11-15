#include <assert.h>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// #define CCD_USE_DOUBLE

#include <stq/gpu/groundtruth.cuh>
#include <stq/gpu/simulation.cuh>
#include <stq/gpu/util.cuh>
#include <stq/gpu/memory.cuh>
// #include <stq/gpu/klee.cuh>
#include <stq/gpu/io.cuh>

#include <spdlog/spdlog.h>

using namespace std;
using namespace stq::gpu;

// spdlog::set_level(spdlog::level::trace);
#ifdef _WIN32
char *optarg = NULL;
int optind = 1;
int getopt(int argc, char *const argv[], const char *optstring) {
  if ((optind >= argc) || (argv[optind][0] != '-') || (argv[optind][0] == 0)) {
    return -1;
  }

  int opt = argv[optind][1];
  const char *p = strchr(optstring, opt);

  if (p == NULL) {
    return '?';
  }
  if (p[1] == ':') {
    optind++;
    if (optind >= argc) {
      return '?';
    }
    optarg = argv[optind];
    optind++;
  }
  return opt;
}
#else
#include <unistd.h>
#endif



bool is_file_exist(const char *fileName) {
  ifstream infile(fileName);
  return infile.good();
}

int main(int argc, char **argv) {
  spdlog::set_level(static_cast<spdlog::level::level_enum>(0));
  vector<char *> compare;

  MemHandler *memhandle = new MemHandler();

  char *filet0;
  char *filet1;

  filet0 = argv[1];
  if (is_file_exist(argv[2]))
    filet1 = argv[2];
  else
    filet1 = argv[1];

  vector<Aabb> boxes;
  Eigen::MatrixXd vertices_t0;
  Eigen::MatrixXd vertices_t1;
  Eigen::MatrixXi faces;
  Eigen::MatrixXi edges;

  parseMesh(filet0, filet1, vertices_t0, vertices_t1, faces, edges);
  constructBoxes(vertices_t0, vertices_t1, edges, faces, boxes);
  size_t N = boxes.size();

  int nbox = 0;
  int parallel = 0;
  bool evenworkload = false;
  int devcount = 1;
  bool pairing = false;
  bool sharedqueue_mgpu = false;
  bool bigworkerqueue = false;

  int memlimit = 0;

  int o;
  while ((o = getopt(argc, argv, "c:n:b:p:d:v:WPQZ")) != -1) {
    switch (o) {
    case 'c':
      optind--;
      for (; optind < argc && *argv[optind] != '-'; optind++) {
        compare.push_back(argv[optind]);
        // compare_mathematica(overlaps, argv[optind]);
      }
      break;
    case 'n':
      N = atoi(optarg);
      break;
    case 'b':
      nbox = atoi(optarg);
      break;
    case 'v':
      memlimit = atoi(optarg);
      break;
    case 'p':
      parallel = stoi(optarg);
      break;
    case 'd':
      devcount = atoi(optarg);
      break;
    case 'W':
      evenworkload = true;
      break;
    case 'P':
      pairing = true;
      break;
    case 'Q':
      sharedqueue_mgpu = true;
      break;
    case 'Z':
      bigworkerqueue = true;
      break;
    }
  }

  vector<pair<int, int>> overlaps;
  int2 *d_overlaps; // device
  int *d_count;     // device
  int tidstart = 0;

  if (evenworkload)
    run_sweep_sharedqueue(boxes.data(), memhandle, N, nbox, overlaps,
                          d_overlaps, d_count, parallel, tidstart, devcount,
                          memlimit);
  // else if (sharedqueue_mgpu)
  //   run_sweep_multigpu_queue(boxes.data(), N, nbox, overlaps, parallel,
  //                            devcount);
  // else if (bigworkerqueue)
  //   run_sweep_bigworkerqueue(boxes.data(), N, nbox, overlaps, d_overlaps,
  //                            d_count, parallel, devcount);
  else
    run_sweep_multigpu(boxes.data(), N, nbox, overlaps, parallel, devcount);

  spdlog::debug("Final CPU overlaps size : {:d}", overlaps.size());

  for (auto i : compare) {
    compare_mathematica(overlaps, i);
  }
}
