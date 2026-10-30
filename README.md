# A Stepwise Encoding Framework for Optimization of Quantum Boolean Circuit Synthesis

This project represents the feasible solution space of quantum Boolean circuit synthesis with a **four-layer stepwise encoding**, and uses metaheuristic algorithms such as AE-QTS to search for circuits with the fewest gates.

The research was first implemented in **Python**. To **improve execution speed** and **handle problems with more bits**, we used **Claude AI** to help convert the Python implementation into **C++** and **CUDA** versions. All three versions share the same encoding, circuit synthesis rules, and fitness function (the total gate count after reduction).

## Versions

| Folder | Language | Purpose | Algorithms |
|---|---|---|---|
| [`python/`](python/) | Python 3 | Original research implementation; easy to read and modify | ID 1–9 |
| [`cpp/`](cpp/) | C++17 | About 120–180× faster than Python on a single thread; runs trials in parallel with multiple threads | ID 1–10 |
| [`cuda/`](cuda/) | C++17 + CUDA | The search runs on the CPU, while circuit synthesis and fitness evaluation run in batches on the GPU; intended for problems of 13 bits and above | ID 1–10 |

| ID | Algorithm | ID | Algorithm |
|---:|---|---:|---|
| 1 | AE-QTS | 6 | TS |
| 2 | QTS | 7 | PSO |
| 3 | QEA | 8 | WOA |
| 4 | GA | 9 | ABC |
| 5 | DE | 10 | AE-QTS-GB (C++/CUDA only) |

The C++ and CUDA versions take the arguments `[num_bits] [problem_idx] [algo_id]`; if they are omitted, the program prompts for them like the Python version. Each version has its own copy of the benchmark library in `data/problems.txt`, and the copies are identical.

## Quick Start

### Python

```bash
cd python
pip install numpy pandas openpyxl
python main.py            # enter the number of bits, problem index and algorithm when prompted
```

### C++

Requires a C++17 compiler and CMake (or g++ directly).

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/qbcs 5 1 1 --experiments 10 --iterations 1000 --seed 2026
```

See [`cpp/README.md`](cpp/README.md) for all options.

## Deploying the CUDA Version

### 1. Requirements

| Item | Requirement |
|---|---|
| GPU | NVIDIA GPU with an up-to-date driver |
| CUDA Toolkit | 12.x or later; RTX 50 series requires 12.8 or later |
| CMake | 3.24 or later |
| Compiler | Windows: Visual Studio 2022 (with "Desktop development with C++") / Linux: a GCC version supported by your CUDA Toolkit |

Check the installation:

```bash
nvidia-smi        # shows the GPU and driver version
nvcc --version    # shows the CUDA Toolkit version
```

### 2. Find Your GPU Architecture

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU:

| GPU | Value |
|---|---|
| RTX 30 series (e.g. 3080, 3090) | `86` |
| RTX 40 series (e.g. 4080, 4090) | `89` |
| RTX 50 series (e.g. 5080, 5090) | `120` |

For other GPUs, see [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus).

### 3. Build

**Windows** (run in "x64 Native Tools Command Prompt for VS 2022"; the example uses an RTX 40 series GPU):

```powershell
cd cuda
cmake -S . -B build -A x64 -DQBCS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release --parallel
```

**Linux / WSL**:

```bash
cd cuda
cmake -S . -B build -DQBCS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 4. Verify GPU Results First

On a new machine, first run a small problem with `--check-gpu` (compares every GPU gate count with the CPU and stops on any mismatch) and `--verify` (checks every circuit against the truth table):

```powershell
# Windows
.\build\Release\qbcs_cuda.exe 5 1 1 --backend cuda --experiments 3 --iterations 20 --threads 3 --seed 2026 --check-gpu --verify --out exp_check
```

```bash
# Linux
./build/qbcs_cuda 5 1 1 --backend cuda --experiments 3 --iterations 20 --threads 3 --seed 2026 --check-gpu --verify --out exp_check
```

### 5. Run Experiments

Once verified, drop `--check-gpu` and `--verify`. For example, 13-bit problem 1 with AE-QTS:

```bash
./build/qbcs_cuda 13 1 1 --backend cuda --experiments 100 --iterations 10000 --threads 8 --gpu-batch 128 --seed 2026 --out exp_cuda
```

Results are written to the `--out` folder (a `.txt` summary and a `.csv` convergence history). Common GPU options:

| Option | Default | Description |
|---|---|---|
| `--backend` | `cuda` | `cuda`, `cpu` (no GPU), or `emulated` (runs the GPU kernel logic on the CPU, for debugging) |
| `--gpu-batch` | 128 | Maximum number of candidate solutions per GPU batch |
| `--gpu-device` | 0 | Which GPU to use on multi-GPU machines |
| `--gpu-memory-mb` | 0 | GPU memory limit in MiB; 0 = use about 80% of free memory |

Without an NVIDIA GPU, build with `-DQBCS_ENABLE_CUDA=OFF` and run with `--backend cpu`. See [`cuda/README.md`](cuda/README.md) for architecture details, memory estimates, multi-GPU runs, and benchmarking.

## License

MIT License; see [`LICENCE`](LICENCE). Copyright (c) 2026 Wei-Chieh Lai and Kuo-Chun Tseng, ASC Lab, National Ilan University.
