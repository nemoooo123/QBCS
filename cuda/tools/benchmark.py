#!/usr/bin/env python3
"""Measure end-to-end CPU versus CUDA wall time with exact result validation.

Example (run on the target NVIDIA GPU, from the project root):
  python tools/benchmark.py --binary build/qbcs_cuda --bits 13 --problem 1 \
      --experiments 32 --iterations 200 --neighbors 10 \
      --threads 1,4,16 --gpu-batches 32,128,512 --gpu-decodes thread,warp

Each measurement includes process startup, GPU setup, computation, and output.
All runs use the same seed and problem.  The complete histories and the global
best gate list must equal the CPU reference before a speedup is reported.
No GPU hardware is needed for --backend emulated, but those measurements are
explicitly labeled emulated and must not be described as CUDA performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from result_compare import compare_results, read_result, result_digest, run_command


def parts(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                            "python": platform.python_version()}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            process = subprocess.run([smi, "--query-gpu=name,uuid,driver_version,memory.total",
                                      "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            info["nvidia_smi"] = process.stdout.strip() if process.returncode == 0 else process.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            info["nvidia_smi"] = str(exc)
    else:
        info["nvidia_smi"] = "not found"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="optional untouched original executable; otherwise use --backend cpu")
    parser.add_argument("--backend", choices=("cuda", "emulated"), default="cuda")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/problems.txt")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--problem", type=int, default=1)
    parser.add_argument("--algorithm", type=int, default=1)
    parser.add_argument("--experiments", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--delta", type=float)
    custom = parser.add_mutually_exclusive_group()
    custom.add_argument("--tt")
    custom.add_argument("--tt-file", type=Path)
    custom.add_argument("--tt-sparse", type=Path)
    parser.add_argument("--threads", default="1,4,16", help="worker counts to compare for both CPU and CUDA")
    parser.add_argument("--gpu-batches", default="32,128,512")
    parser.add_argument("--gpu-decodes", default="thread,warp")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-memory-mb", type=int, default=0)
    parser.add_argument("--gpu-wait-us", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1, help="untimed launches per configuration; warms clocks/caches")
    parser.add_argument("--timeout", type=float, default=3600, help="maximum seconds for one executable invocation")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.data = args.data.resolve()
    if args.baseline:
        args.baseline = args.baseline.resolve()
    for path in (args.binary, args.data, *([args.baseline] if args.baseline else [])):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    threads, batches = list(map(int, parts(args.threads))), list(map(int, parts(args.gpu_batches)))
    decodes = parts(args.gpu_decodes)
    if (not threads or not batches or min(threads + batches) < 1 or args.experiments < 1
            or args.iterations < 1 or args.neighbors < 4 or args.repeats < 1 or args.warmups < 0):
        parser.error("positive counts are required; neighbors >= 4; warmups >= 0")
    if not decodes or any(d not in ("thread", "warp") for d in decodes):
        parser.error("--gpu-decodes must contain thread and/or warp")
    if args.algorithm not in range(1, 11):
        parser.error("algorithm must be in 1..10")
    if len(threads) != len(set(threads)) or len(batches) != len(set(batches)) or len(decodes) != len(set(decodes)):
        parser.error("thread, batch, and decode lists must not contain duplicate configurations")
    if args.backend == "emulated":
        decodes = ["thread"]  # Emulation does not execute the CUDA warp decoder.
    root = args.out.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f"output directory is not empty; choose a new --out path: {root}")
    root.mkdir(parents=True, exist_ok=True)
    custom_args: list[str] = []
    if args.tt is not None:
        custom_args = ["--tt", args.tt]
    elif args.tt_file is not None:
        custom_args = ["--tt-file", str(args.tt_file.resolve())]
    elif args.tt_sparse is not None:
        custom_args = ["--tt-sparse", str(args.tt_sparse.resolve())]
    common = ([str(args.algorithm), *custom_args] if custom_args else
              [str(args.bits), str(args.problem), str(args.algorithm)])
    common += ["--data", str(args.data), "--experiments", str(args.experiments),
               "--iterations", str(args.iterations), "--neighbors", str(args.neighbors),
               "--ga-k", str(min(5, args.neighbors)),
               "--seed", str(args.seed), "--quiet"]
    if args.delta is not None:
        common += ["--delta", repr(args.delta)]
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report: dict[str, Any] = {"settings": settings, "hardware": hardware(), "status": "running",
                              "timing": "process wall time including setup and output; median of measured repeats",
                              "validation": "exact complete histories, all trial final counts, and global best gate list",
                              "records": [], "summary": []}
    if args.backend == "emulated":
        report["warning"] = "Emulated measurements run on CPU and are not CUDA/GPU measurements."
        print(report["warning"], flush=True)

    def checkpoint() -> None:
        (root / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    reference: dict[str, Any] | None = None

    def measure(backend: str, worker_count: int, batch: int, decode: str) -> list[float]:
        nonlocal reference
        label = f"{backend}_t{worker_count}_b{batch}_{decode}"
        durations: list[float] = []
        for run_number in range(-args.warmups, args.repeats):
            phase = f"warmup{-run_number}" if run_number < 0 else f"repeat{run_number + 1}"
            out = root / "runs" / label / phase
            executable = args.baseline if backend == "cpu" and args.baseline else args.binary
            command = [str(executable), *common, "--threads", str(worker_count), "--out", str(out)]
            if not (backend == "cpu" and args.baseline):
                command += ["--backend", backend]
            if backend != "cpu":
                command += ["--gpu-batch", str(batch), "--gpu-decode", decode,
                            "--gpu-device", str(args.gpu_device), "--gpu-memory-mb", str(args.gpu_memory_mb),
                            "--gpu-wait-us", str(args.gpu_wait_us)]
            print(f"Running {label} {phase} ...", flush=True)
            run = run_command(command, out, args.timeout)
            result = read_result(out, expected_trials=args.experiments,
                                 expected_generations=args.iterations)
            if reference is None:
                if backend != "cpu":
                    raise AssertionError("the first benchmark must be the CPU reference")
                reference = result
            else:
                compare_results(reference, result)
            record = {"backend": backend, "threads": worker_count, "gpu_batch": batch,
                      "gpu_decode": decode, "phase": phase, "measured": run_number >= 0,
                      "wall_seconds": run["wall_seconds"], "result_sha256": result_digest(result),
                      "exact_result_match": True}
            report["records"].append(record)
            checkpoint()
            if run_number >= 0:
                durations.append(run["wall_seconds"])
            print(f"  {run['wall_seconds']:.6f}s; exact result match", flush=True)
        return durations

    try:
        cpu: dict[int, float] = {}
        for count in threads:
            times = measure("cpu", count, 0, "none")
            cpu[count] = statistics.median(times)
            report["summary"].append({"backend": "cpu", "threads": count, "gpu_batch": 0,
                                       "gpu_decode": "none", "median_wall_seconds": cpu[count],
                                       "min_wall_seconds": min(times), "max_wall_seconds": max(times),
                                       "speedup_vs_same_threads_cpu": 1.0})
        fastest_cpu = min(cpu.values())
        for count in threads:
            for batch in batches:
                for decode in decodes:
                    times = measure(args.backend, count, batch, decode)
                    median = statistics.median(times)
                    report["summary"].append({"backend": args.backend, "threads": count,
                                               "gpu_batch": batch, "gpu_decode": decode,
                                               "median_wall_seconds": median, "min_wall_seconds": min(times),
                                               "max_wall_seconds": max(times),
                                               "speedup_vs_same_threads_cpu": cpu[count] / median,
                                               "speedup_vs_fastest_tested_cpu": fastest_cpu / median})
                    checkpoint()
        for entry in report["summary"]:
            entry.setdefault("speedup_vs_fastest_tested_cpu", fastest_cpu / entry["median_wall_seconds"])
        report["status"] = "passed"
        report["fastest_tested_cpu_wall_seconds"] = fastest_cpu
        candidates = [item for item in report["summary"] if item["backend"] == args.backend]
        report["best_candidate"] = min(candidates, key=lambda item: item["median_wall_seconds"])
        with (root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["summary"][0]))
            writer.writeheader()
            writer.writerows(report["summary"])
        checkpoint()
        best = report["best_candidate"]
        print(f"\nBest measured {args.backend}: threads={best['threads']}, gpu-batch={best['gpu_batch']}, "
              f"decode={best['gpu_decode']}; median {best['median_wall_seconds']:.6f}s")
        print(f"Speedup versus the fastest tested CPU configuration: "
              f"{best['speedup_vs_fastest_tested_cpu']:.3f}x")
        print("Every reported run matched the CPU reference exactly (timings excluded).")
        print(f"Report and complete logs: {root}")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        checkpoint()
        print(f"Benchmark stopped: {exc}\nLogs retained: {root}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
