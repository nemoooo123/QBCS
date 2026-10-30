// qbcs.h - Quantum Boolean Circuit Synthesis (stepwise-encoding framework), C++17 port
//
// Faithful port of the Python package (utils/init_state.py, utils/topology.py, core/*.py).
// The four encoding layers (Layer 1: cycle priority, Layer 2: break edge, Layer 3: path
// nodes, Layer 4: gate order) are stored as FLAT arrays. The nested Python lists are
// reconstructed through the offsets held in `Encoding`, so no per-individual allocation
// happens in the search loop. Positions that Python marks with the sentinel 999 (Hamming
// distance 1 steps) carry no bits at all here - they are handled structurally instead.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace qbcs {

// ---------------------------------------------------------------------------------------
// Random number generator: xoshiro256++ (public domain, D. Blackman & S. Vigna).
// Replaces Python's random.random()/np.random; sequences are NOT identical to Python
// (impossible across languages) but the algorithms are distribution-equivalent.
// ---------------------------------------------------------------------------------------
class Rng {
public:
    explicit Rng(uint64_t seed);
    uint64_t next();
    // Uniform double in [0, 1), 53 random bits (same as Python's random.random()).
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    // Uniform double in [lo, hi)  (np.random.uniform)
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
    // Uniform integer in [lo, hi] inclusive  (random.randint)
    int randint(int lo, int hi);
    // Uniform integer in [0, n)  (random.randrange / np.random.randint(0, n))
    int below(int n) { return randint(0, n - 1); }
    // k distinct indices drawn uniformly from [0, n)  (random.sample(range(n), k))
    void sample_indices(int n, int k, std::vector<int>& out);

private:
    uint64_t s_[4];
};

// ---------------------------------------------------------------------------------------
// Encoding: everything build_encode() derives from the cycle decomposition of the target
// permutation, plus the flat-array offsets used by Sample / QLayers.
// ---------------------------------------------------------------------------------------
struct Encoding {
    int n = 0;                 // number of qubits (num_bits)
    int num_cycles = 0;

    // Layer 1: one binary code of `l1_bits` bits per cycle (cycle priority)
    int l1_bits = 0;
    int l1_size = 0;           // num_cycles * l1_bits

    // Layer 2: one binary code of l2_bits[c] bits per cycle (break-edge / entry point)
    std::vector<int> l2_bits;  // ceil(log2(cycle_length))
    std::vector<int> l2_off;   // size num_cycles + 1, offsets into the flat Layer-2 array
    int l2_size = 0;
    std::vector<int> mapping;  // node_mapping_index: 1 for 2-cycles, cycle_length otherwise

    // Trajectories: per cycle, the reversed cycle closed by repeating its first node
    // (transition_trajectories in Python), length = cycle_length + 1.
    std::vector<std::vector<int>> traj;
    std::vector<int> nsteps;   // per cycle: 1 for 2-cycles, cycle_length otherwise
    std::vector<int> step_off; // size num_cycles + 1, offsets into the per-step arrays below
    int total_steps = 0;

    // Per step (transition traj[c][s] -> traj[c][s+1]):
    std::vector<int> step_d;   // Hamming distance d
    std::vector<int> step_nb;  // ceil(log2(d)) bits per path node (0 => the "999" marker step)
    // Layer 3: d nodes x step_nb bits per non-marker step
    std::vector<int> l3_off;   // size total_steps + 1
    int l3_size = 0;
    // Layer 4: (2d - 2) direction bits per non-marker step
    std::vector<int> l4_off;   // size total_steps + 1
    int l4_size = 0;

    // Per-cycle offsets into the flat Layer-1 / Layer-3 / Layer-4 arrays (size num_cycles + 1);
    // used by the cycle-granularity crossover of GA/WOA/ABC. (Layer 2 uses l2_off.)
    std::vector<int> l1_cyc_off;
    std::vector<int> l3_cyc_off;
    std::vector<int> l4_cyc_off;

    int layer_size(int layer) const;                 // layer in 0..3
    const std::vector<int>& cyc_off(int layer) const; // per-cycle offsets for layer 0..3
};

// find_cycles(): cycle decomposition of the permutation; 1-cycles (fixed points) dropped.
std::vector<std::vector<int>> find_cycles(const std::vector<int>& perm);

// build_encode(): derive the Encoding from the cycles. `cycles` is consumed as in Python
// (each cycle is reversed and closed).
Encoding build_encoding(std::vector<std::vector<int>> cycles, int num_bits);

// ---------------------------------------------------------------------------------------
// One individual (a "neighbor" in QTS terminology): sampled 0/1 decisions of all layers.
// ---------------------------------------------------------------------------------------
struct Sample {
    std::vector<uint8_t> l1, l2, l3, l4;

    std::vector<uint8_t>& layer(int i);
    const std::vector<uint8_t>& layer(int i) const;
    void resize(const Encoding& E);
    bool operator==(const Sample& o) const { return l1 == o.l1 && l2 == o.l2 && l3 == o.l3 && l4 == o.l4; }
};

// ---------------------------------------------------------------------------------------
// Probability layers (qindividuals1..4 in Python). Each decision bit stores the pair
// (P[bit=0], P[bit=1]) at indices 2*i and 2*i+1, exactly as the Python (k,2) matrices do,
// so the rotation/boundary arithmetic is bit-for-bit identical.
// ---------------------------------------------------------------------------------------
struct QLayers {
    std::vector<double> l1, l2, l3, l4;

    std::vector<double>& layer(int i);
    const std::vector<double>& layer(int i) const;
    void init(const Encoding& E);   // all probabilities 0.5
};

// gen_nbrs(): sample one individual from the probability layers (Layer 4 is repaired).
void sample_individual(const Encoding& E, const QLayers& Q, Rng& rng, Sample& out);

// repair_sequence_logic(): balance the number of 0s and 1s in one Layer-4 step.
void repair_l4_step(uint8_t* bits, int len, Rng& rng);
void repair_l4_step_at(const Encoding& E, std::vector<uint8_t>& l4, int step, Rng& rng);
void repair_l4_all(const Encoding& E, std::vector<uint8_t>& l4, Rng& rng);

// ---------------------------------------------------------------------------------------
// Gates: a multiple-controlled Toffoli = one hypercube edge. `ctrl` holds the control
// pattern (the state with the target bit cleared); `target` is the flipped bit (0 = LSB).
// Python prints a gate as a list of length n with 0/1 controls and 3 at the target.
// ---------------------------------------------------------------------------------------
struct Gate {
    uint32_t ctrl;
    int target;
};
using Circuit = std::vector<Gate>;

std::string circuit_to_python_str(const Circuit& c, int n);  // e.g. [[0, 1, 3], [3, 0, 1]]

// verify_circuit_logic(): simulate the circuit on all 2^n inputs, compare with the target.
bool verify_circuit(const Circuit& c, int n, const std::vector<int>& target);

// ---------------------------------------------------------------------------------------
// Synthesizer: decode_and_synthesize() -> synthesize_route() -> generate_state_trajectory()
// -> assemble_reversible_circuit() with the linear-time A*A=I / moving-rule accumulator.
// Holds scratch buffers, so create one per thread and reuse it.
// ---------------------------------------------------------------------------------------
class Synthesizer {
public:
    explicit Synthesizer(const Encoding& E);

    // Gate count of the reduced circuit for this individual (fitness to minimise).
    int gate_count(const Sample& s);
    // Same, and also writes the reduced circuit (in emission order).
    int synthesize(const Sample& s, Circuit& out);

    const Encoding& encoding() const { return E_; }

private:
    struct RawGate {
        uint32_t v0;   // ctrl pattern (state with target bit cleared)
        int target;
        int prev0, prev1;  // previous live gate touching v0 / v1 (per-vertex stack links)
        bool live;
    };

    void run(const Sample& s);
    void push_edge(uint32_t a, uint32_t b);
    void emit_transposition(const Sample& s, int cycle, int step);

    static constexpr int kMaxBits = 32;

    const Encoding& E_;
    std::vector<int32_t> head_;      // per hypercube vertex: index of the top live gate, -1 if none
    std::vector<RawGate> raw_;
    std::vector<int> order_;         // cycle processing order (Layer 1)
    std::vector<int> cyc_prio_;      // Layer 1 decoded priorities
    int live_ = 0;
};

// ---------------------------------------------------------------------------------------
// Algorithm driver API
// ---------------------------------------------------------------------------------------
enum class Algo { AE_QTS = 1, QTS = 2, QEA = 3, GA = 4, DE = 5, TS = 6, PSO = 7, WOA = 8, ABC = 9 };
const char* algo_name(Algo a);
bool algo_from_id(int id, Algo& out);

struct Params {
    int max_iterations = 10000;
    int num_neighbors = 10;      // population size N
    // AE-QTS / QTS / QEA
    double delta_theta_aeqts = 0.01;
    double delta_theta_qts = 0.01;
    double delta_theta_qea = 0.0005;
    // GA
    int ga_k = 5; double ga_pc = 0.8; double ga_pm = 0.005;
    // DE
    double de_cr = 0.01;
    // TS
    int ts_tabu_size = 45;
    // PSO
    double pso_w = 0.6, pso_c1 = 3.0, pso_c2 = 1.0;
    // WOA
    double woa_b = 0.45, woa_a_start = 2.0, woa_a_end = 0.0, woa_a_decay_power = 1.0;
    // ABC
    int abc_limit = 40;
    // Debug: verify every synthesized circuit against the truth table.
    bool verify = false;
};

struct RunResult {
    int best_gate_count = 0;
    Circuit best_circuit;
    std::vector<double> history;   // best-so-far gate count after each iteration
    long verify_failures = 0;      // only when Params::verify
};

// Run a single experiment (one trial) of the chosen algorithm.
RunResult run_experiment(Algo algo, const Encoding& E, const Params& P, Rng& rng,
                         const std::vector<int>& target /* for --verify */);

// Individual algorithm entry points (all share the same signature).
RunResult run_aeqts(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);
RunResult run_qts  (const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);
RunResult run_qea  (const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);
RunResult run_ga   (const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);
RunResult run_de   (const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);
RunResult run_ts   (const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);
RunResult run_pso  (const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);
RunResult run_woa  (const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);
RunResult run_abc  (const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target);

// Shared helpers used by several algorithms (defined in algorithms_common.cpp)
// Stable ascending sort of indices by fitness (Python: sorted(..., key=gate_count)).
void argsort_stable(const std::vector<int>& fitness, std::vector<int>& idx);
// Evaluate + optionally verify one individual; bumps `failures` on a logic error.
int evaluate(Synthesizer& S, const Sample& s, const Params& P, const std::vector<int>& target,
             long& failures);
// Cycle-granularity single-point crossover of one layer: child = a[:cp cycles] + b[cp:].
void crossover_layer(const Encoding& E, int layer, const Sample& a, const Sample& b, int cp,
                     Sample& child);

}  // namespace qbcs
