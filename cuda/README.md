# QBCS CUDA: A GPU Version That Preserves the Original Search Logic

This project is a rewrite of the Python / C++ quantum Boolean circuit synthesis programs, using an architecture in which **C++ controls the search and CUDA performs batched synthesis and fitness evaluation**. It keeps the original C++ version's 9 algorithms (ID 1–9; the original C++ variants with IDs 10–13 have been removed), four-layer encoding, RNG, Layer 4 repair, ranking, and update order. The added parallelism applies only to mutually independent candidate solutions and independent trials.

This version has been deployed on NVIDIA RTX 4090, RTX 5060, RTX 5080, and RTX 5090 GPUs. Validation details are in `docs/VALIDATION.md`. If CUDA is unavailable at run time the program reports an error; use `--backend cpu` to select the CPU explicitly.

## 1. Quick Start

Requires a C++17 compiler, CMake 3.24 or later, and the NVIDIA CUDA Toolkit. RTX 5060 / 5080 / 5090 have compute capability 12.0, so the build argument is `120`; CUDA Toolkit 12.8 or later and a compatible driver are recommended. The build was tested with CUDA 13.0.88. GPU architectures are listed in the [NVIDIA GPU capability table](https://developer.nvidia.com/cuda/gpus); for Toolkit architecture support see the [NVIDIA CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

### Windows: Visual Studio 2022 x64 Developer Command Prompt / PowerShell

Install Visual Studio's "Desktop development with C++" and CUDA's Visual Studio integration, then go to the `cuda` directory:

```powershell
cmake -S . -B build -A x64 -DQBCS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

First run a small problem, comparing every GPU fitness value with the original CPU synthesizer and checking the truth table:

```powershell
.\build\Release\qbcs_cuda.exe 5 1 1 --backend cuda --experiments 3 --iterations 20 --neighbors 50 --threads 3 --seed 2026 --check-gpu --verify --out exp_check
```

Starting settings for real runs:

```powershell
.\build\Release\qbcs_cuda.exe 13 1 1 --backend cuda --experiments 32 --iterations 200 --neighbors 50 --threads 8 --gpu-batch 128 --seed 2026 --out exp_cuda
```

These are starting settings for measurement. Set `--experiments`, `--iterations`, `--neighbors`, and `--delta` according to your experiment specification; the program adds no extra stopping criteria and does not change the search space.

### Linux / WSL

```bash
cmake -S . -B build -DQBCS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/qbcs_cuda 5 1 1 --backend cuda --experiments 3 --iterations 20 --neighbors 50 --threads 3 --seed 2026 --check-gpu --verify --out exp_check
```

`ctest` runs CPU tests of the shared core and the scheduling service; only the CUDA run command above actually launches the GPU. The executable path differs between Windows and Linux; all other CLI arguments are the same.

For other GPUs, set `CMAKE_CUDA_ARCHITECTURES` to their compute capability, e.g. `89` for an RTX 4090. If it is not set, the project defaults to SASS and PTX for `86`; for PTX compatibility see the [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

### Reference / Verification Build Without CUDA

```bash
cmake -S . -B build-cpu -DQBCS_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
./build-cpu/qbcs_cuda 5 1 1 --backend cpu --experiments 3 --iterations 20 --seed 2026
./build-cpu/qbcs_cuda 5 1 1 --backend emulated --experiments 3 --iterations 20 --seed 2026 --check-gpu
```

`emulated` runs, on the CPU, the integer core shared with the thread-mode CUDA kernel. It is used to locate problems in decoding, packing, memory allocation, and scheduling. It does not simulate GPU performance and does not execute warp intrinsics. The CUDA and CPU builds share the same search drivers.

## 2. What Stays the Same

| Item | Handling in this version |
|---|---|
| Cycle decomposition | Same decomposition, reversal, and closure as the original |
| Layer 1 | MSB-first decoding; priorities in descending order; ties keep the original cycle index order |
| Layer 2 | Keeps `% mapping`; a 2-cycle still uses only its original single step; break-edge selection unchanged |
| Layer 3 | Full encoded search; ties in priority are still ordered by the original MSB-first index |
| Layer 4 | Same sampling, RNG order, and Fisher–Yates repair; same head/tail/pivot gate order |
| Gate reduction | Original `A*A=I` and Moving Rule, applied over the full emission order, including cancellation across cycles |
| Fitness | Reduced gate count, kept as an integer minimization objective |
| Ranking | Stable fitness sort; the pairing order of equal-fitness samples is unchanged |
| AE-QTS update | Each pair is updated in rank order by `delta/(r+1)`, with boundary handling after every single update |
| Best tracking | Original strict improvement, historical best, and each algorithm's acceptance rules |
| Randomness | Original C++ xoshiro256++; same per-trial seeds and random call order |
| Experiment output | Original TXT / CSV and per-generation best-so-far history, plus backend, batch, and per-stage timing information |

Python and C++ already used different RNGs and different sampling call orders. This project therefore uses two kinds of comparison: fixed-sample replay against Python, and same-seed per-generation comparison against the original C++. The latter is the reference trajectory for the CUDA version. Re-run the regression when changing compilers or math libraries, especially for algorithms with more floating-point arithmetic such as PSO. Fast math is not used, and the GPU fitness core uses only integer arithmetic.

### Algorithm IDs

| ID | Name | Core rule preserved |
|---|---|---|
| 1 | AE-QTS | Current-generation harmonic best–worst pairing, direct probability increments |
| 2 | QTS | Current-generation single best–worst, direct probability increments |
| 3 | QEA | Every bit updated toward the historical best |
| 4 | GA | Original tournament / cycle crossover / mutation / elitism |
| 5 | DE | Original binary mutation / crossover / `<=` acceptance and synchronous replacement |
| 6 | TS | Original compound move, tabu list, and aspiration |
| 7 | PSO | Original continuous position / velocity, discretization, and pbest / gbest |
| 8 | WOA | Original in-generation immediate update of population / leader |
| 9 | ABC | Original scout → employed → onlooker order with in-place updates |

**ID 1 in the original program is not an angle-rotation version.** For IDs 1 / 2 / 3, `--delta` is a direct probability step. See `docs/REFERENCE_AUDIT.md` for the full audit.

## 3. Acceleration Architecture

The CPU generates and repairs candidate solutions in the original order. Multiple independent trials hand their independently evaluable samples to a shared scheduling service, which merges the requests into GPU batches. The CPU resolves the L1 / L2 schedule; the GPU decodes L3 / L4 in parallel over `sample × active step`, builds the full unreduced gate sequence, and then performs Moving Rule reduction with one sequential chain per sample. Only the integer fitness values are copied back to the CPU; the final best circuit is rebuilt once by the original CPU synthesizer.

Specific structural improvements:

- Topology metadata is built once and stays resident on the GPU; candidate and scratch buffers are reused.
- The Layer 3 sort key encodes both the priority and the original tie index, so the thread decoder needs only one small `path[]` array, which is reused to store the decoded edges.
- Each raw gate is packed into a 32-bit integer, and each predecessor link in the reducer is also 32-bit. This keeps the data needed to preserve ordering while reducing scratch space.
- Schedule, raw gates, and predecessor buffers use an `[item][batch]` layout, so adjacent GPU lanes working on different samples access memory as contiguously as possible.
- Host staging uses pinned memory and asynchronous CUDA copies; GPU work can overlap with CPU sampling / updates of other trials. The GPU stages of each batch still run in order on one stream.
- If a population exceeds the batch capacity, the service splits it automatically, and the trial updates only after all parts finish. The batch size is not the population size N.

These strategies follow NVIDIA's guidance on reducing transfers, keeping data resident, and memory access in the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html). The design derivation, memory formulas, and performance limits are in `docs/CUDA_DESIGN.md`.

WOA and ABC keep their in-generation dependencies and are evaluated one candidate at a time; only independent requests from different trials are merged. GA / DE / TS / PSO and the quantum family can batch a whole set of independent candidates. This difference may give different algorithms different speedups.

## 4. GPU Options and Memory

| Option | Default | Purpose |
|---|---:|---|
| `--backend` | `cuda` | `cuda`, the original `cpu`, or the shared-core `emulated` |
| `--threads` | number of logical CPU cores, capped at the trial count | Independent CPU trials run concurrently; separate from N |
| `--gpu-batch` | 128 | Maximum candidates per batch; smaller batches run at their actual size |
| `--gpu-memory-mb` | 0 | GPU array budget in MiB; 0 uses about 80% of free VRAM at start-up |
| `--gpu-device` | 0 | Which GPU this process uses |
| `--gpu-wait-us` | 200 | Maximum time to wait for other trials to fill a batch; can be 0 |
| `--gpu-decode` | `thread` | `thread` or `warp`; choose whichever measures faster |
| `--check-gpu` | off | Compare every accelerated fitness with the original CPU fitness |
| `--verify` | off | Rebuild every evaluated circuit and check the full truth table; expensive |
| `--trial-offset` | 0 | Global starting trial index, for splitting one experiment across processes / machines |

The program computes the VRAM each sample needs from the actual topology and lowers the batch capacity automatically to fit the budget. **A smaller batch does not reduce the population N.** The GPU budget covers the main device arrays; the CPU population / probability arrays, pinned staging, CUDA context, and other runtime overhead are additional. For larger problems also watch system RAM: a large `--threads` keeps many trial states in memory at once.

Allocation examples computed directly from the included problem tables:

| Problem | Device arrays per sample | Capacity 128 plus shared metadata |
|---|---:|---:|
| `13 1` / `f13_1` | about 1.46 MiB | about 186.87 MiB |
| `17 1` / `nth_prime17_inc` | about 28.07 MiB | about 3,595.70 MiB |

These come from the allocation formula; they are not measured GPU usage or run times. Requirements for 20-bit problems also depend on the number of cycles and the Hamming distance of each step. The CUDA packed format supports 1–27 state bits, and the upper bound on unreduced gates per candidate must not exceed `2^27−1`; the original int offsets and probability indices also have checked capacity. Exceeding these limits reports an error; the practical problem size is still determined by RAM / VRAM and time.

## 5. Per-Generation Verification on Real Hardware

The C++ version in [`../cpp`](../cpp) is the baseline for per-generation comparison. From the `cuda` directory:

```bash
cmake -S ../cpp -B build-reference -DCMAKE_BUILD_TYPE=Release
cmake --build build-reference --config Release --parallel
python3 tests/regression.py --baseline build-reference/qbcs --candidate build/qbcs_cuda --backends cuda --gpu-decodes thread,warp --work-dir regression_cuda
```

On Windows, change the two executable paths to `build-reference/Release/qbcs.exe` and `build/Release/qbcs_cuda.exe`, and `python3` to `python`.

The full regression covers all 9 algorithms, odd and even N, delta edge cases, identity, distance-1 steps, different thread counts, and splitting across batches. Each comparison checks every trial's full history, each trial's best gate count, and the complete gate list of the global best circuit; timing fields are excluded. See `tests/README.md` for details.

Small problems can additionally be checked with Compute Sanitizer on a GPU:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/qbcs_cuda 5 1 1 --backend cuda --gpu-decode thread --experiments 3 --iterations 5 --threads 3 --seed 2026 --check-gpu --verify --out sanitizer_thread
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/qbcs_cuda 5 1 1 --backend cuda --gpu-decode warp --experiments 3 --iterations 5 --threads 3 --seed 2026 --check-gpu --verify --out sanitizer_warp
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/qbcs_cuda 5 1 1 --backend cuda --gpu-decode warp --experiments 3 --iterations 5 --threads 3 --seed 2026 --check-gpu --out sanitizer_sync
```

## 6. Finding a Faster Configuration

After the GPU correctness checks pass, run:

```bash
python3 tools/benchmark.py --binary build/qbcs_cuda --baseline build-reference/qbcs --bits 13 --problem 1 --algorithm 1 --experiments 32 --iterations 200 --neighbors 50 --threads 1,4,8,16 --gpu-batches 32,128,512 --gpu-decodes thread,warp --out benchmark_13bit
```

Each configuration is warmed up and then timed over 3 full process runs, reporting the median wall time. Every run is compared against the full result for the same seed, and speedups are listed only after the comparison passes, against both the CPU at the same thread count and the fastest CPU setting tested. Outputs are `benchmark_summary.csv`, `benchmark.json`, and the full run logs. Use a new folder for `--out`.

For a first pass, narrow it to `--threads 4,8 --gpu-batches 32,128 --repeats 1 --warmups 0`, then repeat formal measurements around the better region. The GPU summary lists actual batch sizes and per-stage times for H2D+clear / decode / reduce / D2H, which helps identify the next bottleneck.

In an instrumented 13-bit AE-QTS profile of the original C++, synthesis took about 58.4% of sampled CPU time, sampling / repair about 26.3%, and updates about 15.2%. This shows that keeping sampling and updates on the CPU leaves a significant cost, so measure speedups with full wall time rather than estimating them from CUDA core counts, compiler register statistics, or emulated test times.

## 7. Multiple GPUs / Multiple Machines

Each process uses one GPU. Different processes with the same base seed and non-overlapping trial offsets can run the same experiment in parallel. For example, splitting 100 trials into 40 / 30 / 30:

```bash
./build/qbcs_cuda 13 1 1 --experiments 40 --trial-offset 0 --seed 2026 --gpu-device 0 --out part_000_039
./build/qbcs_cuda 13 1 1 --experiments 30 --trial-offset 40 --seed 2026 --gpu-device 1 --out part_040_069
./build/qbcs_cuda 13 1 1 --experiments 30 --trial-offset 70 --seed 2026 --gpu-device 2 --out part_070_099
```

These three processes can be started simultaneously in different terminals. On different machines, use each machine's local GPU index (e.g. 0 on all of them) and keep all other parameters identical. Splitting does not pool VRAM, and trials do not exchange search state. Trial numbers in the CSV are global; each part's mean / standard deviation describes only that part, so recompute overall statistics from all trial rows when combining.

`tests/shard_regression.py` verifies that trial seeds, splitting, and merged results are consistent; formal experiments can use the same splitting scheme.

## 8. Project Index

| Path | Contents |
|---|---|
| `src/cuda_evaluator.cu` | CUDA kernels, pinned buffers, VRAM allocation, per-stage timing |
| `src/cuda_core.h` | Exact integer decoder and reducer shared by CPU and CUDA |
| `src/batch_layout.*` | Topology, L1 / L2 schedule, capacity estimation, and packing |
| `src/evaluation.*` | Multi-trial batch queue, splitting, result delivery, and error propagation |
| `src/quantum_algos.cpp`, `src/baseline_algos.cpp` | Original search logic with safe batch boundaries |
| `src/synth.cpp` | Original CPU synthesizer / final best-circuit output / verification reference |
| `tests/` | Shared core, scheduling, per-generation comparison, Python replay, and sharding tests |
| `tools/benchmark.py` | CPU / CUDA configuration sweep with exact result comparison and full wall time |
| `docs/` | Design derivation, differences between the source versions, validation evidence |

License: MIT; see [`../LICENCE`](../LICENCE). Authors: Kuo-Chun Tseng, Wei-Chieh Lai, Hong-Sheng Wang, Yan-Ting Wu, and Chih-Yu Tsao, ASC Lab, National Ilan University.
