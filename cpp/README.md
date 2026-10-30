# qbcs — Quantum Boolean Circuit Synthesis (Stepwise Encoding + AE-QTS), C++17 Version

This is a C++17 port of the Python implementation in [`../python`](../python). Its features and command-line interface follow the original `main.py`.
All nine algorithms (AE-QTS, QTS, QEA, GA, DE, TS, PSO, WOA, ABC) are ported.
It is about 120–180× faster than Python on a single thread (measured with N = 10, see below), and can also run trials in parallel with multiple threads.

## File Mapping

| C++ | Original Python | Contents |
|---|---|---|
| `src/qbcs.h` | — | Shared types/API: `Encoding`, `Sample`, `QLayers`, `Synthesizer`, `Params`, `run_*` |
| `src/encoding.cpp` | `utils/init_state.py` | `find_cycles`, `build_encode`, `gen_nbrs` (four-layer sampling), `repair_sequence_logic`, RNG (xoshiro256++) |
| `src/synth.cpp` | `utils/topology.py` | `decode_and_synthesize` → `synthesize_route` → `generate_state_trajectory` → `assemble_reversible_circuit` (`_MRAccumulator`: A·A = I cancellation + Moving Rule), `verify_circuit_logic` |
| `src/quantum_algos.cpp` | `core/AE_QTS.py`, `QTS.py`, `QEA.py` | The three share one main loop and differ only in the `updateQ` rule |
| `src/baseline_algos.cpp` | `core/GA.py`, `DE.py`, `TS.py`, `PSO.py`, `WOA.py`, `ABC.py` | The six baselines |
| `src/data_loader.{h,cpp}` + `data/problems.txt` | `utils/data_loader.py` | 82 benchmark truth tables, 3–17 bits (plain text; adding problems needs no recompilation) |
| `src/main.cpp` | `main.py` | CLI, multithreaded trials, `.txt` / `.csv` output |
| `tests/dump_replay.py`, `tests/replay_test.cpp` | — | Bit-exact equivalence test against the Python version (see below) |
| `tools/csv_to_xlsx.py` | — | Optional: converts `.csv` back to the original `.xlsx` layout |

### Data Structures (compared with Python)

Python stores the four encoding layers as nested lists; C++ uses **flat arrays**, with offset tables in `Encoding` that recover the nested structure:

| Layer | Meaning | Python | C++ |
|---|---|---|---|
| Layer 1 | Cycle priority | `nbr1[i]`: `(num_cycles, bits)` ndarray | `Sample::l1`, `l1_cyc_off[c] + bit` |
| Layer 2 | Break edge / entry point | `nbr2[i][c]`: `ceil(log2 len)` bits | `Sample::l2`, `l2_off[c] + bit` |
| Layer 3 | Path node order | `nbr3[i][c][s]`: `d` nodes × `ceil(log2 d)` bits, or `[999]` | `Sample::l3`, `l3_off[g] + node*nb + bit` |
| Layer 4 | Gate order (head / tail) | `nbr4[i][c][s]`: `2d−2` bits, or `[0]` | `Sample::l4`, `l4_off[g] + bit` |

The `999` marker (a step with Hamming distance 1) **takes no bits** in C++; it is handled directly by `step_nb[g] == 0`,
so the search loop does no dynamic allocation. The probability layers in `QLayers` still store two doubles `(P[0], P[1])` per bit,
exactly like Python's `(k, 2)` matrices, so the floating-point results of `updateQ` match bit for bit.

## Building

Requires a C++17 compiler; there are no third-party dependencies.

**Linux / macOS / Windows (MSYS2 MinGW-w64 or WSL)**
```bash
g++ -std=c++17 -O2 -pthread src/*.cpp -o qbcs
```

**Windows (Visual Studio x64 Native Tools Command Prompt)**
```bat
cl /O2 /std:c++17 /EHsc src\*.cpp /Fe:qbcs.exe
```

**CMake (any platform)**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

In our measurements `-O3` was no faster than `-O2`, and `-march=native` was slightly slower, so `-O2` is recommended.

## Running

Same as `main.py`: `qbcs [num_bits] [problem_idx] [algo_id]`. If the positional arguments are omitted, the program prompts for them like Python does.

```bash
./qbcs 5 1 1                                   # 5-bit problem 1, AE-QTS, 100 trials × 10000 generations, N = 50
./qbcs 13 2 1 --threads 16 --seed 2026         # 16 threads; a fixed seed makes results reproducible
./qbcs 8 3 4 --experiments 30 --iterations 2000
./qbcs 1 --tt "1,0,3,2,5,7,4,6"                # custom truth table (the first positional argument is algo_id)
./qbcs 1 --tt-file my_perm.txt
./qbcs 5 1 1 --experiments 3 --iterations 300 --verify   # simulate every synthesized circuit on all 2^n inputs
```

algo_id: 1 AE-QTS, 2 QTS, 3 QEA, 4 GA, 5 DE, 6 TS, 7 PSO, 8 WOA, 9 ABC.

Main options (defaults = `main.py`):

| Option | Default | Description |
|---|---|---|
| `--experiments N` | 100 | Number of trials |
| `--iterations N` | 10000 | Generations per trial |
| `--neighbors N` | 50 | Population / neighborhood size N |
| `--threads N` | all cores | Number of trials run in parallel; **results depend only on `--seed`, not on the thread count** |
| `--seed S` | time-based | Base random seed (each trial's seed is hashed from the seed and the trial number) |
| `--data FILE` | `data/problems.txt` | Problem library file |
| `--out DIR` | `exp` | Output root directory |
| `--verify` | off | Debugging: simulates every circuit and compares it with the truth table; the failure count is included in the report |
| `--delta`, `--ga-tsr/--ga-pc/--ga-pm`, `--de-cr`, `--tabu`, `--pso-w/-c1/-c2`, `--woa-b`, `--abc-limit` | same as Python | Algorithm parameters (see the parameter table in the [main README](../README.md)); `--ga-k K` sets the GA tournament size directly instead of TSR × N |

### Output

```
exp/<n>_bit/<ALGO>_Results/<ALGO>_<n>_<idx>.txt   # summary, best circuit (Python list format), μ ± σ
exp/<n>_bit/<ALGO>_Results/<ALGO>_<n>_<idx>.csv   # trial × generation matrix
```

The `.csv` layout is identical to the DataFrame Python writes to `.xlsx` with pandas:
the first column is the index (`Trial_1`…, `Average_Convergence`, `Std_Deviation`), followed by `Gen_1`…`Gen_N` and `Execution_Time(s)`.
The `Std_Deviation` row uses the sample standard deviation (ddof = 1) like pandas, and the `μ ± σ` in the `.txt` uses ddof = 0 like `np.std`.

To load a result in pandas, use `pd.read_csv(path, index_col=0)` instead of `pd.read_excel(path)`,
or run `python tools/csv_to_xlsx.py exp` to convert a whole directory back to `.xlsx`.

### Adding Problems

Add one line to `data/problems.txt` (no recompilation needed):

```
<num_bits> <problem_idx> <name> : v0 v1 v2 ... v(2^n-1)
```

## Equivalence with the Python Version

**1. Replay test (bit-exact)** — `tests/dump_replay.py` runs AE-QTS / QTS / QEA with **the original Python code**
and dumps each generation's sampled nbr1–4, the circuits Python synthesized, the fitness ranking, and the probability layers after `updateQ`.
`tests/replay_test.cpp` feeds the same samples to the C++ `Synthesizer` and update rules and compares everything bit for bit:

```bash
tests/run_replay.sh ../python 5 1 200 aeqts 7
# REPLAY OK: aeqts n=5 idx=1 iters=200 | circuits identical: 2000 (236462 gates), ...
```

Passed for 3–10 bits and all three algorithms, about 13,000 circuits / 2.4 M gates in total:
gate lists (including order), rankings, and probability layers were all identical, and every circuit passed truth-table simulation.

**2. The six baselines** — random sequences cannot be reproduced across languages, so `--verify` was used to confirm that all circuits are correct,
and convergence curves were compared (5-bit f1, 30 trials × 300 generations): TS Python 85.20 vs C++ 84.27, ABC 98.33 vs 97.67;
the other seven were also within statistical error.

## Performance (single thread, AE-QTS, N = 10)

| Problem | Python | C++ | Speedup |
|---|---|---|---|
| 5-bit | 6.0 ms/gen | 0.05 ms/gen | ~120× |
| 8-bit | 98 ms/gen | 0.7 ms/gen | ~140× |
| 10-bit | 716 ms/gen | 3.9 ms/gen | ~180× |
| 13-bit | (estimated ~8 s/gen) | 47 ms/gen | ~170× |

A full 13-bit experiment (100 trials × 10000 generations, N = 10) takes about 13 hours on one thread and about 1 hour on 16 threads.

All figures in this section were measured with N = 10; the current default is N = 50.

## Intentional Differences from the Python Version

1. `.xlsx` → `.csv` (same layout, see above); `.txt` is UTF-8 (originally UTF-16).
2. The discarded `gen_nbrs + decode_and_synthesize` call at the start of each trial is not kept (it only consumed random numbers and did not affect results).
3. Functions in `topology.py` that were never called — `reduce_sequences_*`, `determine_transition_strategy`, `analyze_bit_differences`,
   `map_next_transition_bits`, `initialize_solution_layer` — and the unused `redundant_coupling_count` computed inside `synthesize_route`
   were not ported.
4. Two small Python bugs that do not affect fitness statistics were fixed: the best circuit returned by `DE.py` came from the stale
   `circuit_solutions` of the initial population (correct gate count, wrong circuit), and `QTS/QEA/PSO/WOA` stored a `(circuit, count)` tuple.
5. Python's no-op operations on the `999` marker (TS flipping Layer 4's `[0]`, GA/WOA/ABC mutating markers) are no-ops in C++ as well,
   but the probability of selecting a marker is preserved, so the distribution of search behavior is unchanged.
6. Multithreading: each trial's seed = hash(`--seed`, trial number), so the same `--seed` gives the same results with any `--threads`.

## License

MIT; see [`../LICENCE`](../LICENCE). Authors: Kuo-Chun Tseng, Wei-Chieh Lai, Hong-Sheng Wang, Yan-Ting Wu, and Chih-Yu Tsao, ASC Lab, National Ilan University.
