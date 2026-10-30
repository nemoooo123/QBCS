# Validation Record and Items Not Yet Measured

Date: 2026-09-23. Sources: the uploaded Python archive and `qbcs_cpp.zip`. Hashes of the source archives / unmodified C++ reference are in `validation/source_manifest.json`.

> Note: this record predates the removal of the ROT family (IDs 11–13), so the counts below include 13 algorithms. After the removal, the regression for IDs 1–10 was re-run (CPU and emulated backends, 1 and 3 threads) and the results still matched the original C++.

## Completed Validation

| Type | Actual result | Conclusion supported |
|---|---|---|
| Real CUDA compilation | NVCC 13.0.88, GNU C++ 13.3, `sm_120`; full compile/link passed | Kernels and host code build with a real CUDA compiler |
| CPU-only build | `QBCS_ENABLE_CUDA=OFF`; Release build / CTest passed | The reference and shared-core verification tools build without CUDA |
| Python fixed-sample replay | 4 traces, 1,630 circuits / 142,840 gates | With the same encoding, gate lists / order, fitness sort, and replay probability updates match |
| Original C++ → new CPU / emulated | 468 full CLI comparisons, 0 failures | Every trial's per-generation output and the global best gate list match for the 13 production algorithms |
| Batch / wait boundaries | 84 additional CLI comparisons, 0 failures | Capacity smaller than the population and different wait / splitting strategies do not change results |
| Trial sharding | All 13 algorithms; 6 trials split into 2+1+3 match | `--trial-offset` keeps the same seed / trial mapping |
| Shared integer core | 42,978 samples / 40,423 batches / 1,350,216 raw gates | The shared scalar decoder / reducer matches the original full circuits |
| Scheduling service | 24 threads, 2,400 requests / 45,472 samples; fault injection passed | Order, capacity, and exception propagation of concurrent merge / split are correct |
| Host memory checks | ASan + UBSan passed for core and service | No out-of-bounds access or undefined behavior detected on the tested host paths |
| Benchmark tooling | 6 CPU / emulated smoke runs gave consistent results | Timing, reading / comparison, and report generation tools work |

Every correctness run among the 552 CLI comparisons used `--verify`, and the emulated runs also used `--check-gpu`. Full JSON results are in `tests/validation/`. Timing fields are excluded from equality checks; each trial's full history, mean / standard deviation, final gate count, and the complete global best gate list in the report are all compared. The original CLI does not output each trial's own best circuit, so no claim is made here about comparing gate lists that are not output.

## Independent Reference for the Shared-Core Tests

`tests/core_test.cpp` uses the uploaded version's `Synthesizer` as one reference; a second reference is a naive reverse scan that rebuilds the full gate list by allowing only disjoint edges to commute before cancelling identical gates. The latter does not use endpoint stacks / packed tokens / predecessor arrays, so it is not just a rewrite of the same implementation as the reducer under test. Full circuits are also checked against the original truth table.

Coverage includes:

- All 40,320 3-bit permutations, including the identity.
- Randomized multi-sample batches for 3 / 4 / 5 / 8 bits.
- Batch strides 1, 2, 7, 17, 33, 35, 384, with batches resized by reusing the layout.
- All-zero and all-one ties in L1 / L3, L2 modulo, distance 1 and the problem's maximum distance.
- For distance 3, all 64 L3 × 6 valid L4 combinations, 384 in total.
- Step / lane decoders run in reverse order to check that parallel work is independent.
- 37,486 gate cancellations actually hit, of which 15,114 were non-adjacent and 3,063 crossed cycles.
- Buffer guards and checks on the end of each sample's valid emissions.

AddressSanitizer / UndefinedBehaviorSanitizer passed for both core and service. The environment's ptrace restrictions prevented LeakSanitizer from starting, so `detect_leaks=0` was set; this record makes no claim that leak checking passed.

## Warp Decoder Review

The warp kernel was compiled by a real NVCC. A read-only source review and an independent host-side formula comparison were also done: 9,825 cases and 250,773 gates, covering d = 1..27, all balanced L4 values for d <= 7, and random / tied priorities for larger d. Route and head / tail emission indices all match the original pointer method.

These host-side formula checks do not execute CUDA shuffle / ballot instructions and cannot replace validation on a real GPU. `--gpu-decode thread|warp` is provided so that the GPU regression can compare both real kernels.

## GPU Evidence Not Yet Collected

The build environment had no usable NVIDIA device / driver. Running `--backend cuda` reported an incompatible CUDA driver/runtime and an unavailable device, and the process exited with a failure status. **There is no measured data for successful GPU kernel execution, passing Compute Sanitizer, or GPU speedup.**

Items to run on a machine with an NVIDIA GPU:

1. `tests/regression.py --backends cuda --gpu-decodes thread,warp`, compared against the unmodified original C++.
2. Split tests with `--gpu-batch 1` and normal batches.
3. Compute Sanitizer memcheck / synccheck.
4. A full wall-time sweep with `tools/benchmark.py`, compared against the fastest CPU configuration tested.

All of these tools and commands are ready; if CUDA execution fails or results differ, they fail explicitly instead of switching to another backend to produce apparently successful results. Windows compilation and execution were also not tested in that Linux environment.
