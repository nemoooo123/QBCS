# CUDA Design and Equivalence

## Goals and Boundaries

The goal is to accelerate synthesis and fitness evaluation of fully encoded circuits on the GPU while preserving the existing search process. The strict standard is: the same C++ RNG state, candidate solutions, probability updates at every step, tie-breaking, acceptance conditions, number of generations, and stopping rule. No path heuristics were added, the search dimension was not reduced, and no algorithm was replaced.

A heterogeneous CPU/GPU architecture is used because the original program has two different kinds of dependency: the Moving Rule reduction of each circuit needs the full emission order, and some WOA / ABC candidates depend on acceptance results earlier in the same generation. CUDA runs the large number of independent synthesis tasks, while the CPU keeps the already-validated search and random number flow.

## 1. Work That Can Be Reordered

Once a candidate solution is fixed, the Layer 3 priority decoding and Layer 4 raw gate generation of each selected transposition read only that step's encoding. As long as each result is placed at the correct position in the original emission order, the order in which the generation work runs does not affect the output. `(sample, active_step)` can therefore be parallelized, and steps omitted by Layer 2 need not be generated.

The number of active steps chosen in each cycle is fixed: 1 for a 2-cycle, and L−1 for any other cycle of length L. The CPU's `BatchLayout::pack` builds the schedule and gate offsets following the original L1 ordering and the original L2 modulo / pointer logic. Each step emits exactly `2d−1` gates, so each kernel task writes to a non-overlapping range.

Each trial's RNG, population, probability state, and best tracking are fully independent. The scheduling service only merges fitness requests and writes results back in the original caller order. When an N-size population is split into several GPU batches, the caller waits for all N fitness values before continuing, so N, the generation barrier, and the update order are unchanged.

## 2. Equivalent Transformation of the Layer 3 Sort

The original differing bits are listed from MSB to LSB; when priorities are equal, the stable sort keeps this order. For a differing bit p, use the key:

`key = (priority << 5) | p`

Sorting the keys in descending order makes the higher bit position win when priorities are equal, which is exactly the original stable tie order. Since `p < 32`, the priority and the position do not overlap. The 5 bits store the target index.

The thread decoder first stores the keys in a single `path[]` array, then overwrites it in place with the hypercube edges formed in priority order. Because a Toffoli edge is the same gate regardless of which endpoint comes first, Layer 4 only needs to choose `path[head]` or `path[tail−1]`, without storing every route vertex or multiple sorting scratch arrays.

The warp decoder gives each lane one node priority and computes ranks with shuffles; equal priorities are ordered by the smaller original index. Each lane builds the route from the prefix XOR of the ranks, then uses ballot to compute the prefix count of Layer 4 zero bits, giving the same edge indices as the sequential head / tail method. All lanes participate in every shuffle / ballot; warp-related branches are decided per step.

Both decoders preserve the output order. The compiler resource report shows a 112-byte local stack for the thread decoder and no local stack for the warp decoder; on the other hand, the warp decoder's output to `[gate][batch]` may have a larger stride. Which is faster must be decided by measurement on real hardware.

## 3. The Moving Rule's Ordering Cannot Be Removed

The original synthesizer uses the most recent still-alive gate at each vertex to decide whether a new gate can pass through the following commuting gates and cancel. A new gate reads the heads of its two endpoints and takes the later one; if the target matches, it is the same hypercube edge, so the pair cancels and the two predecessor heads are restored.

This decision depends on the results of earlier cancellations, so this version still processes gates one at a time in the full emission order. Each sample's reducer is owned by one GPU lane, and samples run in parallel. Reducing each cycle independently and adding the results would miss cancellations across cycles; using unordered atomic pushes would break chronological precedence.

### Replacing the Original Raw Vector Index with an Emission Ordinal

The original `raw_` appends only when a new live gate is added; in this version the chronological token is numbered by the ordinal of each raw emission, including positions that were cancelled immediately and never appended. Both numberings keep exactly the same relative order for the gates that were actually appended. The reducer only compares order and follows predecessor links, and never relies on the numbering being contiguous, so this transformation is equivalent.

The token is defined as:

`token = ((emission_ordinal + 1) << 5) | target`

0 represents an empty head. The target is less than 32, so `max(token0, token1)` is also the later gate; the two endpoints of the same gate have the same token. Each emission's two previous tokens are two uint32 fields in scratch arrays.

If `max_gates <= 2^27−1`, the left shift above does not overflow. A packed edge is `(ctrl << 5) | target`, which requires at most 27 state bits. The host checks these limits during allocation. The final best gate list is rebuilt by the original CPU `Synthesizer`, so routine GPU fitness evaluation needs no live flags and never returns whole best circuits.

## 4. Memory Formula

Let:

- `V = 2^n`: number of truth-table / hypercube vertices.
- `K`: number of active steps per candidate, constant for a fixed topology.
- `B34 = l3_size + l4_size`: bytes of a candidate's L3 / L4.
- `Gmax`: upper bound on raw emissions per candidate.
- `S`: total number of topology steps.
- `B`: the GPU batch capacity actually allocated.

For a cycle longer than 2, any candidate omits exactly one step. That cycle's raw-gate bound is the sum of all `2d−1` values minus the smallest one. For a 2-cycle, use the cost of its only step. Summing the per-cycle bounds gives `Gmax`.

The bytes of the main device arrays per sample are:

`M_sample = B34 + 8K + 12Gmax + 4V + 8`

which are, respectively: input, schedule + offset, packed raw edge + two previous buffers, vertex heads, and emitted count + fitness. The shared step metadata is `24S` bytes (per this version's Step struct definition). The total device allocation is approximately:

`M_device = B × M_sample + S × sizeof(Step)`

The CUDA context, streams/events, and other runtime costs are not included in the array formula above; the auto budget uses only about 80% of the free VRAM at start-up to leave headroom. Pinned host buffers need about `B × (B34 + 8K + 8)` extra, and each concurrent trial additionally keeps its population / probabilities / best solution in regular RAM.

A smaller batch capacity only changes how execution is split; it does not change N or any search parameter. If not even one sample fits, an index might overflow, or allocation fails, the program stops and reports it clearly; it never truncates buffers, approximates fitness, or switches to sampling the truth table.

## 5. Current Performance Limits and Measurement Method

In one instrumented 13-bit AE-QTS profile of the original C++, about 58.4% of sampled CPU time was attributed to synthesis, 26.3% to sampling / repair, and 15.2% to updates. The grouping and the full original log are in `validation/original_cpu_profile.txt`, including limitations such as compiler inlining / gprof attribution.

If this profile is treated as a simplified single-trial model, assuming the original sampling / update cost is completely fixed, GPU synthesis is instantaneous, and there is no transfer overhead, Amdahl's estimate gives an upper bound of about `1 / (1−0.584) ≈ 2.40`. This only illustrates the remaining CPU cost; it is neither a promise of actual speedup nor a strict upper bound: overlap across trials, different n / N / generations, CPU saturation, data transfer, and profiling instrumentation all change the ratios.

Under this version's requirement of keeping the same C++ seed trajectory, the main limits are CPU sampling / repair / updates and the sequential reduction of each sample. Switching to a GPU RNG, merging harmonic updates, or using a different Moving Rule would lose the current basis for exact comparison. Any future GPU-resident sampling / updating would need its own design and validation of RNG stream scheduling, per-pair clamping, and repair order.

This version provides per-batch H2D+clear / decode / reduce / D2H event timing, plus CPU packing time and batch min / mean / max. `benchmark.py` sweeps threads, batch, and decoder using full-process wall time with exact same-seed result comparison, and compares against the fastest CPU configuration tested. If the GPU is slower on a small problem, a speedup below 1 is reported as is.

## 6. Reference Files and Reproducibility

`reference_cpp/` is a copy of the original C++ source (with the extra variants, IDs 10–13, removed to match this project); `validation/source_manifest.json` records the hashes of the original archives and reference files before that change. `src/synth.cpp` still serves as the original full-circuit reference. The changes to the search drivers mainly split independent evaluate calls into sampling and batched evaluation; DE uses a trial population buffer to defer acceptance, but all candidates are still generated from the same original population.

The verified results should be labeled separately: source replay, CPU production driver comparison, shared-core host execution, service concurrency, and NVCC compilation. This evidence supports the ported logic and its buildability; actual GPU execution and performance still need GPU regression / Compute Sanitizer / benchmark runs.
