// encoding.cpp - port of utils/init_state.py (find_cycles, build_encode, gen_nbrs,
// sample_layer_L1..L4, repair_sequence_logic) onto the flat-array representation.
#include "qbcs.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace qbcs {

// ------------------------------------------------------------------ Rng ------------------
static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

Rng::Rng(uint64_t seed) {
    uint64_t x = seed;
    for (auto& v : s_) v = splitmix64(x);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

uint64_t Rng::next() {
    const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
}

int Rng::randint(int lo, int hi) {
    // Unbiased bounded integer (rejection sampling on the high bits).
    const uint64_t range = static_cast<uint64_t>(hi - lo) + 1ULL;
    if (range <= 1) return lo;
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % range);
    uint64_t r;
    do { r = next(); } while (r >= limit);
    return lo + static_cast<int>(r % range);
}

void Rng::sample_indices(int n, int k, std::vector<int>& out) {
    // Partial Fisher-Yates: k distinct values from [0, n).
    out.resize(n);
    for (int i = 0; i < n; ++i) out[i] = i;
    for (int i = 0; i < k; ++i) {
        int j = randint(i, n - 1);
        std::swap(out[i], out[j]);
    }
    out.resize(k);
}

// ------------------------------------------------------------------ helpers ---------------
static int ceil_log2(int x) {  // == math.ceil(math.log2(x)) for x >= 1
    int b = 0;
    while ((1 << b) < x) ++b;
    return b;
}

static int popcount32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int c = 0;
    while (x) { x &= x - 1; ++c; }
    return c;
#endif
}

int Encoding::layer_size(int layer) const {
    switch (layer) {
        case 0: return l1_size;
        case 1: return l2_size;
        case 2: return l3_size;
        default: return l4_size;
    }
}

const std::vector<int>& Encoding::cyc_off(int layer) const {
    switch (layer) {
        case 0: return l1_cyc_off;
        case 1: return l2_off;
        case 2: return l3_cyc_off;
        default: return l4_cyc_off;
    }
}

// ------------------------------------------------------------------ find_cycles ----------
std::vector<std::vector<int>> find_cycles(const std::vector<int>& perm) {
    std::vector<char> visited(perm.size(), 0);
    std::vector<std::vector<int>> cycles;
    for (size_t i = 0; i < perm.size(); ++i) {
        if (visited[i]) continue;
        std::vector<int> cyc;
        size_t cur = i;
        while (!visited[cur]) {
            visited[cur] = 1;
            cyc.push_back(static_cast<int>(cur));
            int nxt = perm[cur];
            if (nxt < 0 || static_cast<size_t>(nxt) >= perm.size())
                throw std::runtime_error("truth table is not a permutation (value out of range)");
            cur = static_cast<size_t>(nxt);
        }
        if (cyc.size() > 1) cycles.push_back(std::move(cyc));
    }
    return cycles;
}

// ------------------------------------------------------------------ build_encode ---------
Encoding build_encoding(std::vector<std::vector<int>> cycles, int num_bits) {
    Encoding E;
    E.n = num_bits;
    E.num_cycles = static_cast<int>(cycles.size());

    // Layer 1: bits needed to rank the cycles
    E.l1_bits = (E.num_cycles == 1) ? 1 : ceil_log2(E.num_cycles);
    E.l1_size = E.num_cycles * E.l1_bits;

    E.l2_bits.resize(E.num_cycles);
    E.l2_off.assign(E.num_cycles + 1, 0);
    E.mapping.resize(E.num_cycles);
    E.traj.resize(E.num_cycles);
    E.nsteps.resize(E.num_cycles);
    E.step_off.assign(E.num_cycles + 1, 0);
    E.l1_cyc_off.assign(E.num_cycles + 1, 0);
    E.l3_cyc_off.assign(E.num_cycles + 1, 0);
    E.l4_cyc_off.assign(E.num_cycles + 1, 0);
    for (int c = 0; c <= E.num_cycles; ++c) E.l1_cyc_off[c] = c * E.l1_bits;

    for (int c = 0; c < E.num_cycles; ++c) {
        std::vector<int>& cyc = cycles[c];
        const int len = static_cast<int>(cyc.size());
        E.mapping[c] = (len == 2) ? 1 : len;
        E.l2_bits[c] = ceil_log2(len);
        E.l2_off[c + 1] = E.l2_off[c] + E.l2_bits[c];

        // i.reverse(); i.append(i[0])  -> closed transition trajectory
        std::reverse(cyc.begin(), cyc.end());
        cyc.push_back(cyc[0]);
        E.traj[c] = cyc;

        E.nsteps[c] = (len == 2) ? 1 : len;
        E.step_off[c + 1] = E.step_off[c] + E.nsteps[c];
    }
    E.l2_size = E.l2_off[E.num_cycles];
    E.total_steps = E.step_off[E.num_cycles];

    E.step_d.resize(E.total_steps);
    E.step_nb.resize(E.total_steps);
    E.l3_off.assign(E.total_steps + 1, 0);
    E.l4_off.assign(E.total_steps + 1, 0);

    for (int c = 0; c < E.num_cycles; ++c) {
        for (int s = 0; s < E.nsteps[c]; ++s) {
            const int g = E.step_off[c] + s;
            const int d = popcount32(static_cast<uint32_t>(E.traj[c][s] ^ E.traj[c][s + 1]));
            const int nb = ceil_log2(d);  // 0 when d == 1  -> Python's [999, 999] marker
            E.step_d[g] = d;
            E.step_nb[g] = nb;
            // Layer 3: d nodes, nb bits each (marker steps contribute nothing)
            E.l3_off[g + 1] = E.l3_off[g] + d * nb;
            // Layer 4: 2d - 2 direction bits (marker steps contribute nothing)
            E.l4_off[g + 1] = E.l4_off[g] + (nb == 0 ? 0 : 2 * d - 2);
        }
        E.l3_cyc_off[c + 1] = E.l3_off[E.step_off[c + 1]];
        E.l4_cyc_off[c + 1] = E.l4_off[E.step_off[c + 1]];
    }
    E.l3_size = E.l3_off[E.total_steps];
    E.l4_size = E.l4_off[E.total_steps];
    return E;
}

// ------------------------------------------------------------------ Sample / QLayers -----
std::vector<uint8_t>& Sample::layer(int i) {
    switch (i) { case 0: return l1; case 1: return l2; case 2: return l3; default: return l4; }
}
const std::vector<uint8_t>& Sample::layer(int i) const {
    switch (i) { case 0: return l1; case 1: return l2; case 2: return l3; default: return l4; }
}
void Sample::resize(const Encoding& E) {
    l1.assign(E.l1_size, 0);
    l2.assign(E.l2_size, 0);
    l3.assign(E.l3_size, 0);
    l4.assign(E.l4_size, 0);
}

std::vector<double>& QLayers::layer(int i) {
    switch (i) { case 0: return l1; case 1: return l2; case 2: return l3; default: return l4; }
}
const std::vector<double>& QLayers::layer(int i) const {
    switch (i) { case 0: return l1; case 1: return l2; case 2: return l3; default: return l4; }
}
void QLayers::init(const Encoding& E) {
    l1.assign(2 * static_cast<size_t>(E.l1_size), 0.5);
    l2.assign(2 * static_cast<size_t>(E.l2_size), 0.5);
    l3.assign(2 * static_cast<size_t>(E.l3_size), 0.5);
    l4.assign(2 * static_cast<size_t>(E.l4_size), 0.5);
}

// ------------------------------------------------------------------ repair ---------------
// repair_sequence_logic(): after sampling/mutation a Layer-4 step must contain as many 0s
// (head moves) as 1s (tail moves); surplus bits are flipped at uniformly random positions.
void repair_l4_step(uint8_t* bits, int len, Rng& rng) {
    if (len <= 1) return;
    int zeros = 0;
    for (int i = 0; i < len; ++i) zeros += (bits[i] == 0);
    const int ones = len - zeros;
    const int imbalance = (zeros - ones) / 2;  // int() truncates toward zero, as in Python
    if (imbalance == 0) return;

    static thread_local std::vector<int> pos;
    pos.clear();
    const uint8_t from = imbalance > 0 ? 0 : 1;
    const int k = imbalance > 0 ? imbalance : (ones - zeros) / 2;
    for (int i = 0; i < len; ++i)
        if (bits[i] == from) pos.push_back(i);
    // random.sample(pos, k): partial Fisher-Yates over the candidate positions
    const int m = static_cast<int>(pos.size());
    for (int i = 0; i < k; ++i) {
        int j = rng.randint(i, m - 1);
        std::swap(pos[i], pos[j]);
        bits[pos[i]] = static_cast<uint8_t>(1 - from);
    }
}

void repair_l4_step_at(const Encoding& E, std::vector<uint8_t>& l4, int step, Rng& rng) {
    const int len = E.l4_off[step + 1] - E.l4_off[step];
    if (len > 1) repair_l4_step(l4.data() + E.l4_off[step], len, rng);
}

void repair_l4_all(const Encoding& E, std::vector<uint8_t>& l4, Rng& rng) {
    for (int g = 0; g < E.total_steps; ++g) repair_l4_step_at(E, l4, g, rng);
}

// ------------------------------------------------------------------ gen_nbrs -------------
void sample_individual(const Encoding& E, const QLayers& Q, Rng& rng, Sample& out) {
    out.resize(E);
    // Layers 1-3: P[0] < r  ->  1   (sample_layer_L1 / L2 / L3)
    for (int i = 0; i < E.l1_size; ++i) out.l1[i] = (Q.l1[2 * i] < rng.uniform()) ? 1 : 0;
    for (int i = 0; i < E.l2_size; ++i) out.l2[i] = (Q.l2[2 * i] < rng.uniform()) ? 1 : 0;
    for (int i = 0; i < E.l3_size; ++i) out.l3[i] = (Q.l3[2 * i] < rng.uniform()) ? 1 : 0;
    // Layer 4: P[0] > r  ->  0, else 1   (sample_layer_L4), then repair every step
    for (int i = 0; i < E.l4_size; ++i) out.l4[i] = (Q.l4[2 * i] > rng.uniform()) ? 0 : 1;
    repair_l4_all(E, out.l4, rng);
}

}  // namespace qbcs
