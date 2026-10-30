// Integer-only synthesis primitives shared by the CUDA backend and the CPU emulator.
// Copyright/attribution: derived from the supplied qbcs C++ port; see LICENCE.
#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define QBCS_HD __host__ __device__ __forceinline__
#else
#define QBCS_HD inline
#endif

namespace qbcs { namespace gpu_core {

// Packed gates use 5 target bits and at most 27 state bits. Tokens use 5 target
// bits and 27 chronological-index bits. These limits are checked on the host.
constexpr int max_state_bits = 27;
constexpr uint32_t max_emissions = (uint32_t(1) << 27) - 1;

struct Step {
    uint32_t start, diff;
    int distance, node_bits, l3_offset, l4_offset;
};

QBCS_HD int low_bit(uint32_t v) {
#ifdef __CUDA_ARCH__
    return __ffs(static_cast<int>(v)) - 1;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(v);
#else
    int p = 0;
    while ((v & 1u) == 0) { v >>= 1; ++p; }
    return p;
#endif
}

QBCS_HD uint32_t edge(uint32_t state, int target) {
    return ((state & ~(uint32_t(1) << target)) << 5) | uint32_t(target);
}

// An order key (priority, bit position) reproduces stable descending priority:
// original indices enumerate differing bits MSB-first, so larger bit positions
// must win a tie. Reuse the same small array for ordered path edges afterwards.
QBCS_HD void decode_step(const Step& s, const uint8_t* l3, const uint8_t* l4,
                         uint32_t* emitted, size_t stride) {
    const int d = s.distance;
    if (s.node_bits == 0) {
        emitted[0] = edge(s.start, low_bit(s.diff));
        return;
    }
    uint32_t path[max_state_bits];
    int node = 0;
    for (int p = max_state_bits - 1; p >= 0; --p) {
        if ((s.diff >> p) & 1u) {
            uint32_t priority = 0;
            for (int b = 0; b < s.node_bits; ++b)
                priority = (priority << 1) | l3[s.l3_offset + node * s.node_bits + b];
            path[node++] = (priority << 5) | uint32_t(p);
        }
    }
    for (int i = 1; i < d; ++i) {
        const uint32_t key = path[i];
        int j = i - 1;
        while (j >= 0 && path[j] < key) { path[j + 1] = path[j]; --j; }
        path[j + 1] = key;
    }
    uint32_t state = s.start;
    for (int i = 0; i < d; ++i) {
        const int t = int(path[i] & 31u);
        path[i] = edge(state, t);
        state ^= uint32_t(1) << t;
    }
    int head = 0, tail = d, out = 0;
    const uint8_t* sequence = l4 + s.l4_offset;
    for (int j = 0; j < d - 1; ++j) {
        const int index = sequence[j] == 0 ? head++ : --tail;
        emitted[size_t(out++) * stride] = path[index];
    }
    emitted[size_t(out++) * stride] = path[head];
    const int old_head = head;
    head = tail; tail = old_head;
    for (int j = d - 1; j < 2 * d - 2; ++j) {
        const int index = sequence[j] == 0 ? head++ : --tail;
        emitted[size_t(out++) * stride] = path[index];
    }
}

// Exact moving-rule accumulator. Each lane owns one sample and follows its
// complete emission order. A token's chronological index is an emission index,
// including cancelled pushes: order comparisons are identical to the original
// append-only raw vector's indices. Cancelled emission slots are never read.
// All pointers are already offset to the sample's column in [item][batch].
QBCS_HD int reduce(const uint32_t* emitted, int count, size_t stride,
                   uint32_t* heads, uint32_t* previous0, uint32_t* previous1) {
    int live = 0;
    for (int i = 0; i < count; ++i) {
        const uint32_t code = emitted[size_t(i) * stride];
        const int target = int(code & 31u);
        const uint32_t v0 = code >> 5;
        const uint32_t v1 = v0 | (uint32_t(1) << target);
        const uint32_t p0 = heads[size_t(v0) * stride];
        const uint32_t p1 = heads[size_t(v1) * stride];
        const uint32_t p = p0 > p1 ? p0 : p1;
        if (p != 0 && int(p & 31u) == target) {
            const size_t prior = size_t((p >> 5) - 1u) * stride;
            heads[size_t(v0) * stride] = previous0[prior];
            heads[size_t(v1) * stride] = previous1[prior];
            --live;
        } else {
            const size_t slot = size_t(i) * stride;
            previous0[slot] = p0;
            previous1[slot] = p1;
            const uint32_t token = ((uint32_t(i) + 1u) << 5) | uint32_t(target);
            heads[size_t(v0) * stride] = token;
            heads[size_t(v1) * stride] = token;
            ++live;
        }
    }
    return live;
}

}} // namespace qbcs::gpu_core
#undef QBCS_HD
