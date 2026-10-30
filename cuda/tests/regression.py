#!/usr/bin/env python3
"""Compare an untouched original C++ executable with CPU/emulated/CUDA backends.

Examples:
  python tests/regression.py --baseline ../qbcs_cpp/qbcs --candidate build/qbcs_cuda
  python tests/regression.py --baseline ../qbcs_cpp/qbcs --candidate build/qbcs_cuda \
      --backends cuda --gpu-decodes thread,warp --work-dir regression_gpu

Requires only Python's standard library.  A CUDA test failure is never replaced
with an emulated/CPU run.  All timings are discarded from equality checks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import time

from result_compare import compare_results, read_result, result_digest, run_command


@dataclass(frozen=True)
class Case:
    name: str
    bits: int = 3
    problem: int = 1
    neighbors: int = 10
    generations: int = 13
    seed: int = 20260923
    table: str | None = None
    extra: tuple[str, ...] = ()


CASES = [
    Case("3bit_even"),
    Case("3bit_odd", neighbors=5, seed=7),
    Case("5bit_even", bits=5, neighbors=4, generations=9, seed=0),
    Case("5bit_odd", bits=5, neighbors=5, generations=10, seed=0xFFFFFFFFFFFFFFFF),
    Case("delta_zero", neighbors=4, generations=9, extra=("--delta", "0")),
    Case("delta_clamp", neighbors=5, generations=11, extra=("--delta", "1")),
    Case("identity", neighbors=4, generations=3, table="0,1,2,3,4,5,6,7"),
    Case("max_hamming_swap", neighbors=5, table="7,1,2,3,4,5,6,0"),
    Case("all_marker_steps", neighbors=4, table="1,0,3,2,5,4,7,6"),
]


def separated(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="untouched original C++ executable")
    parser.add_argument("--candidate", type=Path, required=True, help="new qbcs executable")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/problems.txt")
    parser.add_argument("--backends", default="cpu,emulated", help="comma-separated cpu,emulated,cuda")
    parser.add_argument("--threads", default="1,3", help="candidate worker-thread counts, comma separated")
    parser.add_argument("--gpu-decodes", default="thread,warp", help="decode modes to check for non-CPU backends")
    parser.add_argument("--gpu-batch", type=int, default=17, help="nonmultiple of N intentionally exercises partial batches")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-memory-mb", type=int, default=0)
    parser.add_argument("--gpu-wait-us", type=int, default=200)
    parser.add_argument("--algorithms", default=",".join(map(str, range(1, 10))))
    parser.add_argument("--suite", choices=("smoke", "full"), default="full")
    parser.add_argument("--experiments", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--work-dir", type=Path, help="retain all run files and regression_report.json here")
    parser.add_argument("--report", type=Path, help="also write a compact JSON report here")
    args = parser.parse_args()
    args.baseline, args.candidate, args.data = (p.resolve() for p in (args.baseline, args.candidate, args.data))
    for path in (args.baseline, args.candidate, args.data):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    backends, decodes = separated(args.backends), separated(args.gpu_decodes)
    threads, algorithms = list(map(int, separated(args.threads))), list(map(int, separated(args.algorithms)))
    if not backends or any(b not in ("cpu", "emulated", "cuda") for b in backends):
        parser.error("--backends must contain cpu, emulated, and/or cuda")
    if not decodes or any(d not in ("thread", "warp") for d in decodes):
        parser.error("--gpu-decodes must contain thread and/or warp")
    if not threads or min(threads) < 1 or args.experiments < 1 or args.gpu_batch < 1:
        parser.error("threads, experiments and gpu-batch must be positive")
    if not algorithms or any(a not in range(1, 10) for a in algorithms):
        parser.error("algorithm IDs must be 1..9")
    cases = CASES if args.suite == "full" else CASES[:3]
    temporary = tempfile.TemporaryDirectory(prefix="qbcs-regression-") if args.work_dir is None else None
    root = Path(temporary.name) if temporary else args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {"baseline": str(args.baseline), "candidate": str(args.candidate),
              "backends": backends, "suite": args.suite, "passed": 0, "failed": 0,
              "cases": [], "notes": ["Every trial's full CSV history and final count are compared exactly.",
              "The output's global best gate list is compared, including gate order.",
              "--verify checks every evaluated circuit; timing fields are excluded.",
              "Emulated execution is a CPU check of shared scalar code, not a GPU/warp test.",
              "Emulated always runs once with scalar thread decode; cuda checks the requested decode modes."]}
    started = time.perf_counter()
    for case in cases:
        for algorithm in algorithms:
            case_name = f"{case.name}_algo{algorithm:02d}"
            common = ([str(algorithm), "--tt", case.table] if case.table is not None
                      else [str(case.bits), str(case.problem), str(algorithm)])
            common += ["--data", str(args.data), "--experiments", str(args.experiments),
                       "--iterations", str(case.generations), "--neighbors", str(case.neighbors),
                       "--ga-k", str(min(5, case.neighbors)), "--abc-limit", "1",
                       "--tabu", "2", "--seed", str(case.seed), "--quiet", "--verify", *case.extra]
            baseline_dir = root / case_name / "reference"
            try:
                run_command([str(args.baseline), *common, "--threads", "1", "--out", str(baseline_dir)],
                            baseline_dir, args.timeout)
                baseline = read_result(baseline_dir, expected_trials=args.experiments,
                                       expected_generations=case.generations, require_verification=True)
            except Exception as exc:
                print(f"FAIL {case_name}: reference failed: {exc}", file=sys.stderr, flush=True)
                report["failed"] += 1
                report["cases"].append({"case": case_name, "status": "reference_failed", "error": str(exc)})
                continue
            digest = result_digest(baseline)
            for backend in backends:
                for thread_count in threads:
                    for decode in (["none"] if backend == "cpu" else ["thread"] if backend == "emulated" else decodes):
                        label = f"{backend}_t{thread_count}_{decode}"
                        out = root / case_name / label
                        cmd = [str(args.candidate), *common, "--threads", str(thread_count),
                               "--backend", backend, "--out", str(out)]
                        if backend != "cpu":
                            cmd += ["--gpu-batch", str(args.gpu_batch), "--gpu-device", str(args.gpu_device),
                                    "--gpu-decode", decode, "--gpu-memory-mb", str(args.gpu_memory_mb),
                                    "--gpu-wait-us", str(args.gpu_wait_us), "--check-gpu"]
                        record = {"case": case_name, "backend": backend, "threads": thread_count,
                                  "decode": decode, "reference_sha256": digest}
                        try:
                            run = run_command(cmd, out, args.timeout)
                            candidate = read_result(out, expected_trials=args.experiments,
                                                    expected_generations=case.generations,
                                                    require_verification=True)
                            compare_results(baseline, candidate)
                            record.update(status="passed", wall_seconds=run["wall_seconds"],
                                          candidate_sha256=result_digest(candidate))
                            report["passed"] += 1
                            print(f"PASS {case_name:26} {label}", flush=True)
                        except Exception as exc:
                            record.update(status="failed", error=str(exc))
                            report["failed"] += 1
                            print(f"FAIL {case_name} {label}: {exc}", file=sys.stderr, flush=True)
                        report["cases"].append(record)
            # Save a checkpoint after each original/candidate comparison group.
            (root / "regression_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["wall_seconds"] = time.perf_counter() - started
    payload = json.dumps(report, indent=2) + "\n"
    (root / "regression_report.json").write_text(payload, encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(f"\nRegression: {report['passed']} passed, {report['failed']} failed; "
          f"{report['wall_seconds']:.2f}s", flush=True)
    if not temporary:
        print(f"Logs and report: {root}")
    elif report["failed"]:
        print("Use --work-dir PATH to retain commands and complete logs.")
    if temporary:
        temporary.cleanup()
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
