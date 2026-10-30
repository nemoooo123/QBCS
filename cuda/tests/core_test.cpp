// The CUDA integer primitives are also valid ordinary C++: exercise their actual
// implementation on the host, without needing a CUDA installation or GPU.
//
// Independent oracles:
//   * the supplied, unchanged Synthesizer supplies the complete expected circuit;
//   * a deliberately simple reverse scan applies the moving rule to decoded raw
//     gates, without endpoint stacks, predecessor links, or packed order tokens;
//   * exhaustive truth-table simulation checks the resulting circuit's meaning.
// This test does not replace a CUDA device run / Compute Sanitizer check.
//
// Build from the project directory:
// g++ -std=c++17 -O2 -pthread -I src tests/core_test.cpp src/encoding.cpp \
//     src/synth.cpp src/batch_layout.cpp -o core_test
#include "batch_layout.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using qbcs::BatchLayout;
using qbcs::Circuit;
using qbcs::Encoding;
using qbcs::Gate;
using qbcs::Sample;
using qbcs::Synthesizer;

struct Counters {
    uint64_t permutations = 0;
    uint64_t batches = 0;
    uint64_t samples = 0;
    uint64_t raw_gates = 0;
    uint64_t cancellations = 0;
    uint64_t commuting_cancellations = 0;
    uint64_t cross_cycle_cancellations = 0;
    uint64_t l2_modulo_samples = 0;
    uint64_t distance_one = 0;
    uint64_t distance_maximum = 0;
} stats;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

template <class T> class Guarded {
public:
    Guarded(size_t size, T fill, T guard)
        : size_(size), guard_(guard), buffer_(size + 2 * margin_, guard) {
        std::fill(data(), data() + size_, fill);
    }
    T* data() { return buffer_.data() + margin_; }
    const T* data() const { return buffer_.data() + margin_; }
    T& operator[](size_t i) { return data()[i]; }
    const T& operator[](size_t i) const { return data()[i]; }
    void check(const std::string& name) const {
        for (size_t i = 0; i < margin_; ++i) {
            require(buffer_[i] == guard_, name + ": prefix overwritten");
            require(buffer_[margin_ + size_ + i] == guard_, name + ": suffix overwritten");
        }
    }
private:
    static constexpr size_t margin_ = 8;
    size_t size_;
    T guard_;
    std::vector<T> buffer_;
};

uint32_t pack_gate(const Gate& g) {
    return (g.ctrl << 5) | uint32_t(g.target);
}

Gate unpack_gate(uint32_t code) {
    return Gate{code >> 5, int(code & 31u)};
}

bool share_endpoint(uint32_t left, uint32_t right) {
    const Gate a = unpack_gate(left), b = unpack_gate(right);
    const uint32_t a1 = a.ctrl ^ (uint32_t(1) << a.target);
    const uint32_t b1 = b.ctrl ^ (uint32_t(1) << b.target);
    return a.ctrl == b.ctrl || a.ctrl == b1 || a1 == b.ctrl || a1 == b1;
}

// All-controls Toffoli gates swap just their two incident vertices. Thus two
// different such gates commute exactly when their endpoints are disjoint. Walk
// backwards across these independent gates, cancel an identical gate if found,
// and otherwise append. This uses neither the core's data structures nor its
// latest-token comparison, and preserves the surviving chronological order.
Circuit naive_moving_rule(const std::vector<uint32_t>& raw,
                         const std::vector<int>& cycle) {
    struct Live { uint32_t code; int cycle; };
    std::vector<Live> live;
    for (size_t i = 0; i < raw.size(); ++i) {
        bool cancelled = false;
        for (size_t j = live.size(); j > 0; --j) {
            const size_t k = j - 1;
            if (!share_endpoint(live[k].code, raw[i])) continue;
            if (live[k].code == raw[i]) {
                ++stats.cancellations;
                if (k + 1 != live.size()) ++stats.commuting_cancellations;
                if (live[k].cycle != cycle[i]) ++stats.cross_cycle_cancellations;
                live.erase(live.begin() + k);
                cancelled = true;
            }
            break;
        }
        if (!cancelled) live.push_back(Live{raw[i], cycle[i]});
    }
    Circuit result;
    result.reserve(live.size());
    for (const Live& gate : live) result.push_back(unpack_gate(gate.code));
    return result;
}

std::vector<int> permutation_for_cycles(int n,
                                       const std::vector<std::vector<int>>& cycles) {
    std::vector<int> permutation(size_t(1) << n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<bool> used(permutation.size(), false);
    for (const auto& cycle : cycles) {
        require(cycle.size() >= 2, "test cycle must have at least two vertices");
        for (size_t i = 0; i < cycle.size(); ++i) {
            const int v = cycle[i];
            require(v >= 0 && size_t(v) < used.size() && !used[v], "invalid test cycle");
            used[v] = true;
            permutation[v] = cycle[(i + 1) % cycle.size()];
        }
    }
    return permutation;
}

// Construct valid samples without using the production sampler or L4 repair:
// exactly half of every L4 segment is zero, then shuffle the balanced segment.
Sample random_sample(const Encoding& e, std::mt19937_64& rng) {
    Sample s;
    s.resize(e);
    for (int layer = 0; layer < 3; ++layer)
        for (auto& bit : s.layer(layer)) bit = uint8_t(rng() & 1u);
    for (int step = 0; step < e.total_steps; ++step) {
        const int begin = e.l4_off[step], end = e.l4_off[step + 1];
        std::fill(s.l4.begin() + begin + (end - begin) / 2, s.l4.begin() + end, 1);
        std::shuffle(s.l4.begin() + begin, s.l4.begin() + end, rng);
    }
    return s;
}

// L1 / L3 use tied extreme priorities. L2 uses extreme binary codes (including
// codes above a non-power-of-two cycle length); L4 stays structurally valid.
Sample extreme_sample(const Encoding& e, int ones, int l4_pattern) {
    Sample s;
    s.resize(e);
    for (int layer = 0; layer < 3; ++layer)
        std::fill(s.layer(layer).begin(), s.layer(layer).end(), uint8_t(ones));
    for (int g = 0; g < e.total_steps; ++g) {
        const int begin = e.l4_off[g], end = e.l4_off[g + 1], half = (end - begin) / 2;
        for (int j = 0; j < end - begin; ++j) {
            s.l4[begin + j] = uint8_t(l4_pattern == 0 ? j >= half :
                                     l4_pattern == 1 ? j < half : j % 2);
        }
    }
    return s;
}

void check_batch(const std::vector<int>& target, BatchLayout& layout,
                 Synthesizer& oracle, const std::vector<Sample>& samples,
                 const std::string& label) {
    const Encoding& e = layout.encoding;
    const size_t batch = samples.size();
    require(batch > 0, label + ": empty test batch");
    const uint32_t untouched = 0xeeeeeeeeu;
    Guarded<uint8_t> input(layout.input_bytes * batch, 0xcd, 0xda);
    Guarded<int> schedule(size_t(layout.active_steps) * batch, -77, -123456);
    Guarded<int> offsets(size_t(layout.active_steps) * batch, -77, -123456);
    Guarded<int> counts(batch, -77, -123456);
    Guarded<uint32_t> raw(size_t(layout.max_gates) * batch, untouched, 0xdeadbeefu);
    Guarded<uint32_t> heads(layout.vertices * batch, 0, 0xdeadbeefu);
    Guarded<uint32_t> previous0(size_t(layout.max_gates) * batch, untouched, 0xdeadbeefu);
    Guarded<uint32_t> previous1(size_t(layout.max_gates) * batch, untouched, 0xdeadbeefu);
    std::vector<const Sample*> pointers;
    for (const Sample& s : samples) pointers.push_back(&s);
    layout.pack(pointers, input.data(), schedule.data(), offsets.data(), counts.data());

    // The GPU decodes steps independently. Reverse both dimensions to verify
    // that a step never relies on a preceding step or another sample's writes.
    for (int k = layout.active_steps - 1; k >= 0; --k) {
        for (size_t jj = batch; jj > 0; --jj) {
            const size_t i = jj - 1, index = size_t(k) * batch + i;
            const int g = schedule[index], offset = offsets[index];
            require(g >= 0 && g < e.total_steps, label + ": invalid scheduled step");
            require(offset >= 0 && offset + 2 * e.step_d[g] - 1 <= counts[i],
                    label + ": invalid raw offset");
            const uint8_t* row = input.data() + i * layout.input_bytes;
            qbcs::gpu_core::decode_step(layout.steps[g], row, row + e.l3_size,
                                       raw.data() + size_t(offset) * batch + i, batch);
        }
    }

    std::vector<int> cycle_of_step(e.total_steps);
    for (int c = 0; c < e.num_cycles; ++c)
        std::fill(cycle_of_step.begin() + e.step_off[c],
                  cycle_of_step.begin() + e.step_off[c + 1], c);
    for (size_t i = 0; i < batch; ++i) {
        const std::string context = label + ", sample " + std::to_string(i);
        require(counts[i] >= 0 && counts[i] <= layout.max_gates,
                context + ": emission bound violated");
        std::vector<uint32_t> stream(counts[i]);
        std::vector<int> origin(counts[i], -1);
        int end = 0;
        for (int k = 0; k < layout.active_steps; ++k) {
            const size_t index = size_t(k) * batch + i;
            const int g = schedule[index];
            require(offsets[index] == end, context + ": gap or overlap in emitted steps");
            end += 2 * e.step_d[g] - 1;
            std::fill(origin.begin() + offsets[index], origin.begin() + end, cycle_of_step[g]);
            stats.distance_one += (e.step_d[g] == 1);
            stats.distance_maximum += (e.step_d[g] == e.n);
        }
        require(end == counts[i], context + ": incomplete step schedule");
        for (int k = 0; k < counts[i]; ++k) {
            const uint32_t code = raw[size_t(k) * batch + i];
            require(code != untouched, context + ": unwritten raw gate");
            const Gate gate = unpack_gate(code);
            require(gate.target < e.n && gate.ctrl < layout.vertices &&
                        !(gate.ctrl & (uint32_t(1) << gate.target)),
                    context + ": invalid gate encoding");
            stream[k] = code;
        }
        for (int k = counts[i]; k < layout.max_gates; ++k)
            require(raw[size_t(k) * batch + i] == untouched,
                    context + ": wrote beyond this sample's emission count");

        const int count = qbcs::gpu_core::reduce(raw.data() + i, counts[i], batch,
                                                heads.data() + i, previous0.data() + i,
                                                previous1.data() + i);
        const Circuit actual = naive_moving_rule(stream, origin);
        Circuit expected;
        const int expected_count = oracle.synthesize(samples[i], expected);
        require(count == expected_count && size_t(count) == actual.size(),
                context + ": reduced counts differ (core " + std::to_string(count) +
                    ", oracle " + std::to_string(expected_count) + ", scan " +
                    std::to_string(actual.size()) + ")");
        for (size_t k = 0; k < actual.size(); ++k)
            require(pack_gate(actual[k]) == pack_gate(expected[k]),
                    context + ": complete gate sequence differs at gate " + std::to_string(k));
        require(qbcs::verify_circuit(actual, e.n, target),
                context + ": circuit does not realize the target permutation");

        bool modulo = false;
        for (int c = 0; c < e.num_cycles; ++c) {
            int value = 0;
            for (int b = e.l2_off[c]; b < e.l2_off[c + 1]; ++b)
                value = 2 * value + samples[i].l2[b];
            modulo = modulo || value >= e.mapping[c];
        }
        stats.l2_modulo_samples += modulo;
        ++stats.samples;
        stats.raw_gates += stream.size();
    }
    input.check(label + " input");
    schedule.check(label + " schedule");
    offsets.check(label + " offsets");
    counts.check(label + " counts");
    raw.check(label + " raw");
    heads.check(label + " heads");
    previous0.check(label + " previous0");
    previous1.check(label + " previous1");
    ++stats.batches;
}

void exhaustive_three_bit() {
    std::vector<int> target(8);
    std::iota(target.begin(), target.end(), 0);
    std::mt19937_64 rng(0x13458ec23dULL);
    do {
        const Encoding e = qbcs::build_encoding(qbcs::find_cycles(target), 3);
        BatchLayout layout(e);
        Synthesizer oracle(e);
        // One fixed, reproducible valid sample for every permutation, identity
        // included. Other tests cover many samples of the same encoding.
        check_batch(target, layout, oracle, {random_sample(e, rng)},
                    "all 3-bit permutations #" + std::to_string(stats.permutations));
        ++stats.permutations;
    } while (std::next_permutation(target.begin(), target.end()));
    require(stats.permutations == 40320, "did not enumerate all 8! permutations");
}

void targeted_batches() {
    const std::vector<std::pair<int, std::vector<std::vector<int>>>> cases = {
        {1, {}}, {1, {{0, 1}}},
        {3, {}}, {3, {{0, 1}}}, {3, {{0, 7}}},
        {3, {{0, 3, 7}, {1, 2, 6}, {4, 5}}},
        {4, {{0, 15}, {1, 14}, {2, 13}, {3, 12}, {4, 11}, {5, 10}, {6, 9}, {7, 8}}},
        {5, {{0, 31, 3}, {1, 30, 7, 16, 10}, {2, 29}, {4, 5}}},
        {8, {{0, 255}, {1, 254, 17}, {2, 253, 18, 64, 127}, {3, 252}}}
    };
    std::mt19937_64 rng(0xf21a8bccULL);
    for (size_t problem = 0; problem < cases.size(); ++problem) {
        const int n = cases[problem].first;
        const auto target = permutation_for_cycles(n, cases[problem].second);
        const Encoding e = qbcs::build_encoding(qbcs::find_cycles(target), n);
        BatchLayout layout(e);
        Synthesizer oracle(e);
        std::vector<Sample> extremes;
        for (int ones = 0; ones <= 1; ++ones)
            for (int pattern = 0; pattern < 3; ++pattern)
                extremes.push_back(extreme_sample(e, ones, pattern));
        check_batch(target, layout, oracle, extremes, "targeted extremes #" + std::to_string(problem));
        // Reuse the layout with growing and shrinking, non-power-of-two widths.
        for (size_t width : {size_t(33), size_t(1), size_t(17), size_t(2), size_t(7)}) {
            std::vector<Sample> samples;
            for (size_t i = 0; i < width; ++i) samples.push_back(random_sample(e, rng));
            check_batch(target, layout, oracle, samples,
                        "targeted varying stride #" + std::to_string(problem));
        }
    }
}

void random_batches() {
    std::mt19937_64 rng(0xcafefaceda7aULL);
    for (int n : {3, 4, 5, 8}) {
        std::vector<int> target(size_t(1) << n);
        std::iota(target.begin(), target.end(), 0);
        for (int problem = 0; problem < 12; ++problem) {
            std::shuffle(target.begin(), target.end(), rng);
            const Encoding e = qbcs::build_encoding(qbcs::find_cycles(target), n);
            BatchLayout layout(e);
            Synthesizer oracle(e);
            std::vector<Sample> samples;
            for (int i = 0; i < 35; ++i) samples.push_back(random_sample(e, rng));
            // In the same batch, force complete tie sets beside random samples.
            samples[0] = extreme_sample(e, 0, 0);
            samples[1] = extreme_sample(e, 1, 1);
            check_batch(target, layout, oracle, samples,
                        "random n=" + std::to_string(n) + " problem=" + std::to_string(problem));
        }
    }
}

// A distance-three transposition has 2^6 L3 codes and C(4,2) balanced L4
// sequences. Enumerating their product explicitly covers all priority ties and
// every valid head/tail sequence for that distance.
void exhaustive_small_transposition() {
    const auto target = permutation_for_cycles(3, {{0, 7}});
    const Encoding e = qbcs::build_encoding(qbcs::find_cycles(target), 3);
    require(e.l3_size == 6 && e.l4_size == 4, "unexpected distance-three encoding");
    BatchLayout layout(e);
    Synthesizer oracle(e);
    std::vector<Sample> samples;
    for (unsigned int code = 0; code < 64; ++code) {
        std::vector<uint8_t> directions{0, 0, 1, 1};
        do {
            Sample s;
            s.resize(e);
            s.l1[0] = uint8_t(code & 1u);
            s.l2[0] = uint8_t((code >> 1) & 1u);
            for (int i = 0; i < e.l3_size; ++i) s.l3[i] = uint8_t((code >> i) & 1u);
            s.l4 = directions;
            samples.push_back(std::move(s));
        } while (std::next_permutation(directions.begin(), directions.end()));
    }
    require(samples.size() == 384, "did not enumerate all L3/L4 combinations");
    check_batch(target, layout, oracle, samples, "exhaustive distance-three L3/L4");
}
} // namespace

int main() {
    try {
        targeted_batches();
        exhaustive_small_transposition();
        exhaustive_three_bit();
        random_batches();
        require(stats.distance_one > 0 && stats.distance_maximum > 0,
                "missing Hamming-distance boundary coverage");
        require(stats.l2_modulo_samples > 0, "missing L2 modulo coverage");
        require(stats.commuting_cancellations > 0, "missing non-adjacent moving-rule cancellation");
        require(stats.cross_cycle_cancellations > 0, "missing cross-cycle cancellation");
        std::cout << "CORE OK: all " << stats.permutations << " 3-bit permutations; "
                  << stats.samples << " samples in " << stats.batches << " batches; "
                  << stats.raw_gates << " raw gates.\n"
                  << "Complete reduced gate sequences and counts match the supplied C++ oracle; "
                     "every truth table is verified.\n"
                  << "Moving-rule cancellations: " << stats.cancellations
                  << " (non-adjacent " << stats.commuting_cancellations
                  << ", across cycles " << stats.cross_cycle_cancellations << ").\n"
                  << "L2 modulo samples: " << stats.l2_modulo_samples
                  << "; Hamming-distance-1 steps: " << stats.distance_one
                  << "; maximum-distance steps: " << stats.distance_maximum << ".\n"
                  << "Host execution of shared CUDA core only; device execution is a separate check.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CORE FAILED: " << error.what() << '\n';
        return 1;
    }
}
