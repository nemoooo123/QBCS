# Reference Versions and Semantic Audit

This record is based on the Python and C++ archives provided by the user and on the replay tests actually run for this port. Below, `Python/` and `C++/` refer to the root directories of the two original projects. Line numbers refer to the originally uploaded versions.

## Criteria for the Port

The CUDA version uses the algorithms and sampling behavior of the uploaded C++ version as its execution reference. The Python version is used to cross-check the four-layer encoding, circuit synthesis, fitness ranking, and probability updates. The two original versions already differ in their random number generators and in the order in which random numbers are consumed, so the same seed was never guaranteed to produce the same full search trajectory across them.

Three things must be checked separately:

1. **Deterministic results for the same four-layer sample**: the circuit gate list and its order, and the reduced gate count, must be identical. Functional equivalence with a different gate count does not satisfy the porting requirement.
2. **Search results for the same C++ seed**: keep the RNG call order, fitness ranking, tie handling, and update arithmetic, so that the best gate count and best sample of every generation do not change because of batched evaluation.
3. **Circuit function**: output circuits must implement the input truth table. This check complements the first two and cannot by itself prove that the search algorithm is equivalent.

## Python → Original C++ Replay Results

The uploaded C++ project's existing `tests/dump_replay.py` and `tests/replay_test.cpp` were used: the original Python project generated the samples, and the original C++ version replayed them. All tests used population size 10, seed 123, and problem index 1.

| Problem | Algorithm | Generations | Identical circuits | Gates compared | Result |
|---|---|---:|---:|---:|---|
| 3 bit | AE-QTS | 100 | 1,000 | 18,440 | Passed |
| 5 bit | QTS | 30 | 300 | 37,250 | Passed |
| 5 bit | QEA | 30 | 300 | 37,312 | Passed |
| 8 bit | AE-QTS | 3 | 30 | 49,838 | Passed |
| **Total** | | | **1,630** | **142,840** | **All passed** |

Each test confirmed that:

- The gate list synthesized from the same sample, including gate order and every control / target bit, is identical.
- Every synthesized circuit passes full truth-table simulation.
- The sample indices sorted by fitness are identical.
- The four layers of double-precision probability arrays are identical after every generation's update.

**Scope limits:** these are fixed-sample replays of Python against the original C++, not results on a real GPU. The existing `replay_test.cpp` contains its own copy of the probability update functions for replay and does not call the full algorithm drivers, so the CUDA project's driver regression and GPU differential tests are still needed. This table does not mean that the full Python/C++ flow of every algorithm has been verified bit for bit.

Code used for the replay: `C++/tests/dump_replay.py:70–101`, `C++/tests/replay_test.cpp:79–125`.

## Four-Layer Encoding and Synthesis Specification

| Step | Behavior that must be preserved | Original code location |
|---|---|---|
| Cycle decomposition | Walk the permutation in increasing input index; exclude fixed points; reverse each cycle, then close it end to start | `Python/utils/init_state.py:6–36, 83–107`; `C++/src/encoding.cpp:95–150` |
| Layer 1 | Each cycle is encoded as an MSB-first binary priority and sorted by priority in **stable descending** order; ties keep the original cycle index order; a single cycle still keeps one bit | `Python/utils/init_state.py:51–60`; `Python/utils/topology.py:35–37, 90–92`; `C++/src/synth.cpp:139–150` |
| Layer 2 | Decode the entry point from `ceil(log2(cycle_length))` bits, then take it modulo the mapping; a 2-cycle has mapping 1, so its bit exists but always decodes to 0 | `Python/utils/init_state.py:83–96`; `Python/utils/topology.py:39–45`; `C++/src/encoding.cpp:136–148` |
| Transition selection | For a cycle of length L, take L−1 transitions cyclically starting from the entry point; a 2-cycle takes only one transition | `Python/utils/topology.py:98–120`; `C++/src/synth.cpp:152–165` |
| Layer 3 | A step with Hamming distance d has d priorities of `ceil(log2(d))` bits each; sort the priorities in stable descending order, then flip the differing bit positions in **MSB-first** order | `Python/utils/topology.py:587–626`; `C++/src/synth.cpp:75–109` |
| Layer 4 | For d > 1, keep 2d−2 head/tail decisions; after repair there are d−1 zeros and d−1 ones; emit 2d−1 raw gates in the order first half, central pivot, swap pointers, second half | `Python/utils/init_state.py:353–392`; `Python/utils/topology.py:723–757`; `C++/src/synth.cpp:112–131` |
| d = 1 | Emit one gate directly; Python's `999` sentinel is represented in C++ by a zero-length Layer 3/4 range | `C++/src/encoding.cpp:159–175`; `C++/src/synth.cpp:70–72` |
| Gate representation | Each gate is exactly one hypercube edge: one target, and every other qubit has a 0 or 1 control; the target's integer bit index has LSB = 0, and the output list is MSB first | `C++/src/qbcs.h:127–135`; `C++/src/synth.cpp:190–203` |
| Fitness | Number of gates still alive after the whole circuit passes the Deletion Rule / Moving Rule; smaller is better; no other weighted terms | `Python/utils/topology.py:65–67, 759–792`; `C++/src/synth.cpp:175–186` |

The adjacency continuity data computed in `Python/utils/topology.py:121–163` does not contribute to the return value or the fitness. Functions in that file such as `reduce_sequences_*` and `determine_transition_strategy` are not on the current call path either. The port must not re-enable these old functions or add path heuristics.

### Moving Rule and Ordering Constraints on Cancellation

The accumulator records the most recent still-alive gate at each vertex. For the two endpoints of a new edge, it takes the gate with the larger index from the two vertex stacks; if that gate has the same target as the new gate, it is the same edge, so the pair cancels and the previous links of both endpoints are restored. Otherwise the new gate is pushed onto the stacks.

Basis: `Python/utils/topology.py:628–702`, `C++/src/synth.cpp:33–60`.

This state is order-dependent, and cancellation can cross transition and cycle boundaries. The raw gates of each step can be prepared in parallel, but they must be fed into each sample's accumulator in the original full emission order. It is not valid to reduce each cycle separately and add up the per-cycle gate counts as the fitness, nor to look only for the previous gate with the same name while ignoring gates in between that touch the endpoints.

## Sampling, Updates, and Same-Seed Risks

### Differences Already Present Between the Original Python and C++

- Python uses `random`, and some baselines also use the NumPy RNG; C++ uses xoshiro256++ initialized with splitmix64. The same seed does not correspond to the same random stream.
- Python's `gen_nbrs` samples in **layer → population** order; the C++ driver samples in **individual → layer** order. See `Python/utils/init_state.py:180–198`, `C++/src/quantum_algos.cpp:160–166`.
- Python repairs each Layer 4 step immediately after sampling it; C++ `sample_individual` samples all Layer 4 bits first and then repairs step by step. Python's d = 1 Layer 4 sentinel also consumes one unused threshold; C++ has no such bit. See `Python/utils/init_state.py:321–344`, `C++/src/encoding.cpp:242–250`.
- Python AE-QTS, QTS, and QEA sample and synthesize one extra generation before the main loop and then discard it; C++ does not keep this. See `Python/core/AE_QTS.py:31–36`, `QTS.py:32–37`, `QEA.py:38–43`.
- C++ fixed the wrapping / stale-data problems in the best circuit returned by some Python algorithms, and switched output to CSV / UTF-8. These are also described in items 1–6 of "Intentional Differences from the Python Version" in the C++ README (`../cpp/README.md`).

None of this means CUDA may change the C++ RNG order freely; the C++-compatible path of this port must preserve the existing C++ behavior.

### Details That Must Be Preserved

- **Layers 1–3 choose 1 when `P[0] < r`; Layer 4 chooses 0 when `P[0] > r`.** The two behave differently on equality, so the comparison operators must not be unified. `r` is a 53-bit double in [0, 1).
- Layer 4 repair picks positions to flip uniformly and without repetition among the majority bits; the number picked is half the difference between the counts of zeros and ones. It must not be changed to fixed-position repair or independent resampling.
- Fitness uses a stable ascending sort; equal gate counts keep population index order. Even when a pair of individuals has equal fitness, the original program still updates wherever their decision bits differ; such pairs must not be skipped.
- The global best is replaced only on **strict improvement `<`**; a new sample with an equal score must not overwrite the old one.
- For AE-QTS, the pairs for each bit must be updated in order t = 0 to `floor(N/2)−1`; each pair uses `delta/(t+1)`, and **boundary correction is applied immediately after every pair update**. The harmonic updates must not be summed first and clamped only once.
- Different decision bits in the four layers can be processed in parallel, but the pair update order for the same bit must be preserved. Keep the two doubles P[0] and P[1] and the existing order of additions and subtractions; switching to a single probability, float, or fast math may change the trajectory.
- Keep the fixed iteration limit, where the per-generation best is recorded, and the number of samples per generation. The port adds no entropy-based stopping, restarts, adaptive population, or extra local search.

Basis: `C++/src/encoding.cpp:19–56, 209–250`; `C++/src/algorithms_common.cpp:34–37`; `C++/src/quantum_algos.cpp:82–111, 160–225`.

### Quantum-Inspired Algorithms with Similar Names but Different Rules

| ID | Name | Actual update rule |
|---:|---|---|
| 1 | AE-QTS | Current generation's t-th best against t-th worst; probabilities adjusted directly by `delta/(t+1)` |
| 2 | QTS | Current generation's single best against single worst; probabilities adjusted directly by delta |
| 3 | QEA | Every bit adjusted by delta toward the historical global best |

Although IDs 1, 2, and 3 use the variable name `delta_theta`, they are not amplitude rotations. Go by what the code computes, and do not substitute angle formulas from research documents into the probability versions. The original C++ also had four extra variants, AE-QTS-GB (ID 10) and the amplitude-rotation (ROT) family (IDs 11–13), which have been removed from this project.

## Safe Boundaries for Fitness Batching

`evaluate` consumes no random numbers. For candidates that were originally independent, generating all samples first in the original order and then computing fitness in a batch preserves sampling and update behavior. `evaluate_many` is used only in the following places.

| Algorithm | Batched at | Dependencies preserved |
|---|---|---|
| All quantum IDs | All N samples of each generation | All fitness values are completed before the stable sort and the global-best and probability updates |
| GA | Initial population, the N children of each generation | Tournaments still use the same old population; the stable truncation after adding the elite to N+1 is kept |
| DE | Initial population, all trial samples of each generation | Uses an extra trial buffer; all trials are generated from the same old population, then accepted by the original `<=` condition and replaced synchronously |
| TS | The N candidates of each generation | All moves are formed first, then tabu / aspiration checks run in sorted order |
| PSO | Initial population, all particles of each generation | Position/velocity computation for the generation is kept; pbest / gbest are updated only after fitness completes |
| WOA | Initial population | Candidate generation, acceptance, and population / leader updates within each generation are still done one at a time |
| ABC | Initial population | The in-place update order scout → employed → onlooker and the budget are kept |

WOA updates the population immediately after accepting each candidate, and updates the leader immediately if the best improves; later candidates read this new data. ABC's employed and onlooker phases also read the previously updated population, while the onlooker roulette weights are computed once before the phase starts. Generating all candidates of a generation from the old population and accepting them together would change these two algorithms. Basis: `C++/src/baseline_algos.cpp:410–431, 479–513`.

## Capacity and Implementation Checks

- The raw gate buffer must hold the `2d−1` gates of every selected transition; it must not be sized by the final reduced gate count.
- Raw indices / previous links must be separate for each sample and must not share state across samples or trials.
- Vertex heads must be cleared after each sample; the original C++ clears only the touched endpoints. When CUDA clears the whole block or uses another equivalent method, it must also avoid leaving heads from the previous sample.
- Capacity calculations, offset multiplications, and allocation sizes for large problems must use sufficient width and check for overflow; the fact that a gate target fits in 32 bits does not mean memory can actually support a 32-qubit truth table.
- Verifying the GPU core requires at least a CPU/GPU gate-count comparison on the same samples; if the kernel provides gate lists, the emission order should also be compared. Approximate statistical averages alone cannot rule out ordering or memory-corruption problems.

These constraints serve the requirement that "the overall logic and objective do not change"; performance gains should come from batching, data layout, parallel decoding, and equivalent restructuring of the code.
