#pragma once

#include <spdlog/spdlog.h>
#include <cuda/semaphore>

namespace stq::gpu {

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    spdlog::error("GPUassert: {} {} {:d}", cudaGetErrorString(code), file,
                  line);
    if (abort)
      exit(code);
  }
}

class CCDConfig {
public:
  double co_domain_tolerance; // tolerance of the co-domain
  unsigned int mp_start;
  unsigned int mp_end;
  int mp_remaining;
  long unit_size;
  double toi;
  cuda::binary_semaphore<cuda::thread_scope_device> mutex;
  bool use_ms;
  bool allow_zero_toi;
  int max_iter;
  int overflow_flag;
};

class Singleinterval {
public:
  double first;
  double second;
};

class MP_unit {
public:
  Singleinterval itv[3];
  int query_id;
};

class CCDData {
public:
  double v0s[3];
  double v1s[3];
  double v2s[3];
  double v3s[3];
  double v0e[3];
  double v1e[3];
  double v2e[3];
  double v3e[3];
  double ms;     // minimum separation
  double err[3]; // error bound of each query, calculated from each scene
  double tol[3]; // domain tolerance to help decide which dimension to split
#ifdef CCD_TOI_PER_QUERY
  double toi;
  int aid;
  int bid;
#endif
  int nbr_checks = 0;
};

__device__ __host__ struct MemHandler {

  size_t MAX_OVERLAP_CUTOFF = 0;
  size_t MAX_OVERLAP_SIZE = 0;
  size_t MAX_UNIT_SIZE = 0;
  size_t MAX_QUERIES = 0;
  int realcount = 0;
  int limitGB = 0;

  size_t __getAllocatable() {
    size_t free;
    size_t total;
    gpuErrchk(cudaMemGetInfo(&free, &total));

    size_t used = total - free;

    size_t defaultAllocatable = 0.95 * free;
    size_t tmp = static_cast<size_t>(limitGB) * 1073741824;
    size_t userAllocatable = tmp - used;

    spdlog::debug("Can allocate ({:d}) {:.2f}% of memory", userAllocatable,
                  static_cast<float>(userAllocatable) / total * 100);

    return std::min(defaultAllocatable, userAllocatable);
    ;
  }

  void handleNarrowPhase(int &nbr) {
    size_t allocatable = 0;
    size_t constraint = 0;
    allocatable = __getAllocatable();
    nbr = std::min((size_t)nbr, MAX_QUERIES);
    constraint =
      sizeof(CCDData) * nbr + sizeof(CCDConfig); //+ nbr * sizeof(int2) * 2 +
                                                 //+ sizeof(CCDData) * nbr;
    if (allocatable <= constraint) {
      MAX_QUERIES = (allocatable - sizeof(CCDConfig)) / sizeof(CCDData);
      spdlog::debug(
        "Insufficient memory for queries, shrinking queries to {:d}",
        MAX_QUERIES);
      nbr = std::min((size_t)nbr, MAX_QUERIES);
      return;
    }
    constraint =
      sizeof(MP_unit) * nbr + sizeof(CCDConfig) + sizeof(CCDData) * nbr;

    if (allocatable <= constraint) {
      MAX_UNIT_SIZE =
        (allocatable - sizeof(CCDConfig)) / (sizeof(CCDData) + sizeof(MP_unit));
      spdlog::debug(
        "[MEM INITIAL ISSUE]:  unit size, shrinking unit size to {:d}",
        MAX_UNIT_SIZE);
      nbr = std::min(static_cast<size_t>(nbr), MAX_UNIT_SIZE);
      return;
    }
    // we are ok if we made it here

    size_t available_units = allocatable - constraint;
    available_units /= sizeof(MP_unit);
    // size_t default_units = MAX_UNIT_SIZE ? 2 * MAX_UNIT_SIZE : 2 *
    // size_t default_units = 2 * MAX_QUERIES;
    // if we havent set max_unit_size, set it
    spdlog::debug("unit options: available {:d} or overlap mulitplier {:d}",
                  available_units, MAX_UNIT_SIZE);
    size_t default_units = 2 * nbr;
    MAX_UNIT_SIZE = std::min(available_units, default_units);
    spdlog::debug("[MEM INITIAL OK]: MAX_UNIT_SIZE={:d}", MAX_UNIT_SIZE);
    nbr = std::min((size_t)nbr, MAX_UNIT_SIZE);
    return;
  }

  void handleOverflow(int &nbr) {
    size_t constraint = sizeof(MP_unit) * 2 * MAX_UNIT_SIZE +
                        sizeof(CCDConfig) //+ tmp_nbr * sizeof(int2) * 2 +
                        + sizeof(CCDData) * nbr;

    size_t allocatable = __getAllocatable();
    if (allocatable > constraint) {
      MAX_UNIT_SIZE *= 2;
      spdlog::debug("Overflow: Doubling unit_size to {:d}", MAX_UNIT_SIZE);
    } else {
      while (allocatable <= constraint) {
        MAX_QUERIES /= 2;
        nbr = std::min(static_cast<size_t>(nbr), MAX_QUERIES);
        spdlog::debug("Overflow: Halving queries to {:d}", MAX_QUERIES);
        constraint = sizeof(MP_unit) * MAX_UNIT_SIZE +
                     sizeof(CCDConfig) //+ tmp_nbr * sizeof(int2) * 2 +
                     + sizeof(CCDData) * nbr;
      }
    }
    nbr =
      std::min(MAX_UNIT_SIZE, std::min(static_cast<size_t>(nbr), MAX_QUERIES));
  }

  size_t __getLargestOverlap() {
    size_t allocatable = __getAllocatable();
    return (allocatable - sizeof(CCDConfig)) /
           (sizeof(CCDData) + 3 * sizeof(int2));
  }

  void setOverlapSize() { MAX_OVERLAP_SIZE = __getLargestOverlap(); }

  void handleBroadPhaseOverflow(int desired_count) {
    size_t allocatable = __getAllocatable();
    size_t largest_overlap_size = __getLargestOverlap();

    MAX_OVERLAP_SIZE =
      std::min(largest_overlap_size, static_cast<size_t>(desired_count));
    spdlog::info(
      "Setting MAX_OVERLAP_SIZE to {:.2f}% ({:d}) of allocatable memory",
      static_cast<float>(MAX_OVERLAP_SIZE) * sizeof(int2) / allocatable * 100,
      MAX_OVERLAP_SIZE);

    if (MAX_OVERLAP_SIZE < desired_count) {
      MAX_OVERLAP_CUTOFF *= 0.5;
      spdlog::debug(
        "Insufficient memory to increase overlap size, shrinking cutoff 0.5x to {:d}",
        MAX_OVERLAP_CUTOFF);
    }
  }
};

extern MemHandler *memhandle;

// cudaMallocManaged(&memhandle, sizeof(MemHandler));

} // namespace stq::gpu
