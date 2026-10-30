// synth.cpp - port of utils/topology.py: decode_and_synthesize, synthesize_route,
// generate_state_trajectory, assemble_reversible_circuit (+ _MRAccumulator), and
// verify_circuit_logic.
//
// The vestigial "adjacency continuity fitness" loop of synthesize_route (its result was
// never used) and the never-called helpers (reduce_sequences_*, determine_transition_strategy,
// analyze_bit_differences, map_next_transition_bits, initialize_solution_layer) are not ported.
#include "qbcs.h"

#include <algorithm>
#include <stdexcept>

namespace qbcs {

static inline int lowest_bit(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#else
    int i = 0;
    while (!(x & 1u)) { x >>= 1; ++i; }
    return i;
#endif
}

Synthesizer::Synthesizer(const Encoding& E) : E_(E) {
    if (E.n < 1 || E.n > kMaxBits)
        throw std::runtime_error("number of qubits must be between 1 and 32");
    head_.assign(static_cast<size_t>(1) << E.n, -1);
    order_.resize(E.num_cycles);
    raw_.reserve(1 << 12);
}

// _MRAccumulator.push(): gate = hypercube edge {a, b}. Look up the most recently emitted
// still-live gate touching either endpoint; everything emitted after it commutes with the
// new gate, so if it is the very same edge the pair cancels (A*A = I), otherwise the new
// gate becomes live. Per-vertex stacks are kept as linked lists through prev0/prev1.
inline void Synthesizer::push_edge(uint32_t a, uint32_t b) {
    const int t = lowest_bit(a ^ b);
    const uint32_t v0 = a & ~(1u << t);
    const uint32_t v1 = v0 | (1u << t);
    const int p0 = head_[v0];
    const int p1 = head_[v1];
    const int p = p0 > p1 ? p0 : p1;
    if (p >= 0 && raw_[p].target == t) {
        // Same edge: only one edge per target dimension passes through a vertex, so sharing
        // a vertex and the target bit means identical gate -> cancel. p is then on top of
        // both vertex stacks (see topology.py), so both heads roll back to p's predecessors.
        RawGate& g = raw_[p];
        g.live = false;
        head_[v0] = g.prev0;
        head_[v1] = g.prev1;
        --live_;
    } else {
        const int k = static_cast<int>(raw_.size());
        raw_.push_back(RawGate{v0, t, p0, p1, true});
        head_[v0] = k;
        head_[v1] = k;
        ++live_;
    }
}

// generate_state_trajectory() + the per-transposition part of assemble_reversible_circuit()
inline void Synthesizer::emit_transposition(const Sample& s, int cycle, int step) {
    const int g = E_.step_off[cycle] + step;
    const int start = E_.traj[cycle][step];
    const int end = E_.traj[cycle][step + 1];
    const int d = E_.step_d[g];
    const int nb = E_.step_nb[g];

    if (nb == 0) {  // Hamming distance 1 ("999" marker): a single gate
        push_edge(static_cast<uint32_t>(start), static_cast<uint32_t>(end));
        return;
    }

    // --- Layer 3: order in which the differing bits are flipped ---------------------
    // node codes -> priority values; stable sort of indices by value descending
    // (Python: sorted(enumerate(vals), key=val, reverse=True) keeps ties in index order).
    // d <= n <= 32, so a stable insertion sort on fixed arrays beats std::stable_sort
    // (which allocates a temporary buffer on every call).
    int prio_val[kMaxBits], prio_idx[kMaxBits], diff_pos[kMaxBits], route[kMaxBits + 1];
    const uint8_t* bits = s.l3.data() + E_.l3_off[g];
    for (int i = 0; i < d; ++i) {
        int v = 0;
        for (int b = 0; b < nb; ++b) v = (v << 1) | bits[i * nb + b];  // MSB-first
        prio_val[i] = v;
        prio_idx[i] = i;
    }
    for (int i = 1; i < d; ++i) {
        const int key = prio_idx[i];
        const int kv = prio_val[key];
        int j = i - 1;
        while (j >= 0 && prio_val[prio_idx[j]] < kv) { prio_idx[j + 1] = prio_idx[j]; --j; }
        prio_idx[j + 1] = key;
    }

    // differing bit positions listed MSB-first (Python list index 0 = MSB)
    {
        int k = 0;
        const uint32_t diff = static_cast<uint32_t>(start ^ end);
        for (int i = E_.n - 1; i >= 0; --i)
            if ((diff >> i) & 1u) diff_pos[k++] = i;
    }

    route[0] = start;
    int cur = start;
    for (int i = 0; i < d; ++i) {
        cur ^= (1 << diff_pos[prio_idx[i]]);
        route[i + 1] = cur;
    }
    // route has d + 1 nodes: route[0] = start ... route[d] = end

    // --- Layer 4: head/tail emission order of the 2d-1 gates --------------------------
    // 0 = take the next edge from the head side, 1 = from the tail side. The first d-1
    // bits build one side, the pivot edge joins the two pointers, and the remaining d-1
    // bits retrace the first half (this is why each step must be 0/1 balanced).
    const uint8_t* seq = s.l4.data() + E_.l4_off[g];
    const int total = 2 * d - 2;
    const int mid = d - 1;
    int head = 0, tail = d;
    for (int j = 0; j < mid; ++j) {
        if (seq[j] == 0) { push_edge(route[head], route[head + 1]); ++head; }
        else             { push_edge(route[tail - 1], route[tail]); --tail; }
    }
    if (tail - head != 1)
        throw std::runtime_error("Synchronization Error: Path pointers failed to meet at central pivot.");
    push_edge(route[head], route[tail]);
    std::swap(head, tail);
    for (int j = mid; j < total; ++j) {
        if (seq[j] == 0) { push_edge(route[head], route[head + 1]); ++head; }
        else             { push_edge(route[tail - 1], route[tail]); --tail; }
    }
}

// decode_and_synthesize() for one individual + synthesize_route()
void Synthesizer::run(const Sample& s) {
    raw_.clear();
    live_ = 0;

    // --- Layer 1: cycle priorities, processed in descending order (stable) -------------
    const int nc = E_.num_cycles;
    cyc_prio_.resize(nc);
    for (int c = 0; c < nc; ++c) {
        int v = 0;
        const uint8_t* b = s.l1.data() + c * E_.l1_bits;
        for (int i = 0; i < E_.l1_bits; ++i) v = (v << 1) | b[i];
        cyc_prio_[c] = v;
        order_[c] = c;
    }
    std::stable_sort(order_.begin(), order_.end(),
                     [&](int x, int y) { return cyc_prio_[x] > cyc_prio_[y]; });

    for (int c : order_) {
        // --- Layer 2: entry point (which edge of the closed cycle is left out) ---------
        int ptr = 0;
        const uint8_t* b = s.l2.data() + E_.l2_off[c];
        for (int i = 0; i < E_.l2_bits[c]; ++i) ptr = (ptr << 1) | b[i];
        ptr %= E_.mapping[c];  // val % limit if val >= limit else val

        int total = E_.nsteps[c];
        if (total == 1) total = 2;
        for (int k = 0; k < total - 1; ++k) {
            if (ptr >= total) ptr = 0;
            emit_transposition(s, c, ptr);
            ++ptr;
        }
    }

    // reset the vertex stacks for the next call (only touched vertices)
    for (const RawGate& g : raw_) {
        head_[g.v0] = -1;
        head_[g.v0 | (1u << g.target)] = -1;
    }
}

int Synthesizer::gate_count(const Sample& s) {
    run(s);
    return live_;
}

int Synthesizer::synthesize(const Sample& s, Circuit& out) {
    run(s);
    out.clear();
    out.reserve(live_);
    for (const RawGate& g : raw_)
        if (g.live) out.push_back(Gate{g.v0, g.target});
    return live_;
}

// ------------------------------------------------------------------ printing / verify ----
std::string circuit_to_python_str(const Circuit& c, int n) {
    std::string out = "[";
    for (size_t k = 0; k < c.size(); ++k) {
        if (k) out += ", ";
        out += "[";
        for (int i = n - 1; i >= 0; --i) {  // MSB first, like bin(x).zfill(n)
            if (i != n - 1) out += ", ";
            if (i == c[k].target) out += "3";
            else out += ((c[k].ctrl >> i) & 1u) ? "1" : "0";
        }
        out += "]";
    }
    out += "]";
    return out;
}

bool verify_circuit(const Circuit& c, int n, const std::vector<int>& target) {
    const uint32_t full = (1u << n) - 1u;
    for (uint32_t x = 0; x <= full; ++x) {
        uint32_t st = x;
        for (const Gate& g : c) {
            const uint32_t mask = full & ~(1u << g.target);
            if ((st & mask) == g.ctrl) st ^= (1u << g.target);
        }
        if (static_cast<int>(st) != target[x]) return false;
    }
    return true;
}

}  // namespace qbcs
