#pragma once
#include "qbcs.h"
#include "cuda_core.h"
#include <vector>

namespace qbcs {

// Immutable topology, constructed once per problem. CUDA receives only step
// metadata; the sequential CPU RNG and all four original search layers survive.
class BatchLayout {
public:
    explicit BatchLayout(const Encoding& encoding);
    const Encoding& encoding;
    std::vector<gpu_core::Step> steps;
    int active_steps = 0;
    int max_gates = 0;
    size_t vertices = 0;
    size_t input_bytes = 0; // L3 + L4, per sample
    size_t bytes_per_sample() const;
    size_t metadata_bytes() const;

    // samples.size() is the current matrix stride, not the allocated capacity.
    // Input is sample-major. Schedule/offset/raw/scratch matrices are item-major.
    void pack(const std::vector<const Sample*>& samples, uint8_t* input,
              int* schedule, int* offsets, int* gate_counts);
private:
    std::vector<int> cycle_order_, priorities_;
};

} // namespace qbcs
