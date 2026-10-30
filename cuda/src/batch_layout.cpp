#include "batch_layout.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace qbcs {

BatchLayout::BatchLayout(const Encoding& e) : encoding(e) {
    if (e.n < 1 || e.n > gpu_core::max_state_bits)
        throw std::runtime_error("CUDA/emulated packed synthesis supports 1..27 state bits; use --backend cpu for larger inputs.");
    vertices = size_t(1) << e.n;
    input_bytes = size_t(e.l3_size) + size_t(e.l4_size);
    steps.resize(e.total_steps);
    uint64_t bound = 0;
    for (int c = 0; c < e.num_cycles; ++c) {
        int minimum = std::numeric_limits<int>::max();
        for (int s = 0; s < e.nsteps[c]; ++s) {
            const int g = e.step_off[c] + s;
            const int d = e.step_d[g];
            if (d < 1 || d > e.n)
                throw std::runtime_error("invalid step distance in Encoding");
            steps[g] = {uint32_t(e.traj[c][s]), uint32_t(e.traj[c][s] ^ e.traj[c][s + 1]),
                        d, e.step_nb[g], e.l3_off[g], e.l4_off[g]};
            const int cost = 2 * d - 1;
            bound += uint64_t(cost);
            minimum = std::min(minimum, cost);
        }
        if (e.nsteps[c] == 1) ++active_steps;
        else { active_steps += e.nsteps[c] - 1; bound -= uint64_t(minimum); }
    }
    if (bound > gpu_core::max_emissions)
        throw std::runtime_error("per-sample gate bound exceeds 27-bit chronological token capacity; use --backend cpu.");
    max_gates = int(bound);
    cycle_order_.resize(e.num_cycles);
    priorities_.resize(e.num_cycles);
}

size_t BatchLayout::bytes_per_sample() const {
    // input + two schedule matrices + packed edges + two predecessor matrices
    // + dense heads + returned count + actual emitted count.
    return input_bytes + size_t(active_steps) * 2 * sizeof(int)
        + size_t(max_gates) * 3 * sizeof(uint32_t)
        + vertices * sizeof(uint32_t) + 2 * sizeof(int);
}

size_t BatchLayout::metadata_bytes() const { return steps.size() * sizeof(gpu_core::Step); }

void BatchLayout::pack(const std::vector<const Sample*>& samples, uint8_t* input,
                       int* schedule, int* offsets, int* gate_counts) {
    const Encoding& e = encoding;
    const size_t batch = samples.size();
    for (size_t i = 0; i < batch; ++i) {
        const Sample& s = *samples[i];
        for (int l = 0; l < 4; ++l)
            if (s.layer(l).size() != size_t(e.layer_size(l)))
                throw std::runtime_error("Sample layer size does not match Encoding");
        uint8_t* row = input + i * input_bytes;
        if (e.l3_size) std::memcpy(row, s.l3.data(), size_t(e.l3_size));
        if (e.l4_size) std::memcpy(row + e.l3_size, s.l4.data(), size_t(e.l4_size));
        for (int c = 0; c < e.num_cycles; ++c) {
            int value = 0;
            for (int b = 0; b < e.l1_bits; ++b)
                value = (value << 1) | s.l1[c * e.l1_bits + b];
            priorities_[c] = value;
            cycle_order_[c] = c;
        }
        // The secondary index makes tie behaviour explicit, even with std::sort.
        std::sort(cycle_order_.begin(), cycle_order_.end(), [&](int a, int b) {
            return priorities_[a] != priorities_[b] ? priorities_[a] > priorities_[b] : a < b;
        });
        int k = 0, emitted = 0;
        for (int c : cycle_order_) {
            int pointer = 0;
            for (int b = 0; b < e.l2_bits[c]; ++b)
                pointer = (pointer << 1) | s.l2[e.l2_off[c] + b];
            pointer %= e.mapping[c];
            const int total = e.nsteps[c] == 1 ? 2 : e.nsteps[c];
            for (int j = 0; j < total - 1; ++j) {
                if (pointer >= total) pointer = 0;
                const int g = e.step_off[c] + pointer++;
                schedule[size_t(k) * batch + i] = g;
                offsets[size_t(k++) * batch + i] = emitted;
                emitted += 2 * e.step_d[g] - 1;
            }
        }
        if (k != active_steps || emitted > max_gates)
            throw std::runtime_error("internal schedule bound failure");
        gate_counts[i] = emitted;
    }
}

} // namespace qbcs
