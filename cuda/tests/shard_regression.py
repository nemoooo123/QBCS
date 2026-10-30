#!/usr/bin/env python3
"""Check --trial-offset: split trials must reproduce one unsplit invocation.

The original executable has no --trial-offset option, so the unsharded run is
first compared to it, then reconstructed from three uneven new-executable
shards. This verifies reproducible device/job partitioning without combining
the optimization state or changing any trial's random stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from result_compare import compare_results, read_result, run_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/problems.txt")
    parser.add_argument("--backend", choices=("cpu", "emulated", "cuda"), default="emulated")
    parser.add_argument("--algorithms", default=",".join(map(str, range(1, 10))))
    parser.add_argument("--gpu-decode", choices=("thread", "warp"), default="warp")
    parser.add_argument("--gpu-batch", type=int, default=3)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    binary, baseline, data = [p.resolve() for p in (args.candidate, args.baseline, args.data)]
    for path in (binary, baseline, data):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    algorithms = [int(value) for value in args.algorithms.split(",")]
    if not algorithms or any(a not in range(1, 10) for a in algorithms):
        parser.error("algorithm IDs must be 1..9")
    temporary = tempfile.TemporaryDirectory(prefix="qbcs-shards-") if args.work_dir is None else None
    root = Path(temporary.name) if temporary else args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {"backend": args.backend, "passed": 0, "failed": 0, "records": []}
    try:
        for algorithm in algorithms:
            common = ["5", "1", str(algorithm), "--data", str(data), "--neighbors", "5", "--iterations", "11",
                      "--seed", "18446744073709551615", "--abc-limit", "1",
                      "--verify", "--quiet"]
            acceleration = ["--backend", args.backend]
            if args.backend != "cpu":
                acceleration += ["--gpu-batch", str(args.gpu_batch), "--gpu-decode", args.gpu_decode,
                                 "--gpu-device", str(args.gpu_device), "--check-gpu"]

            def launch(label: str, experiments: int, offset: int, original: bool = False):
                out = root / f"algo{algorithm:02d}" / label
                command = [str(baseline if original else binary), *common, "--experiments", str(experiments),
                           "--threads", "1" if original else "3", "--out", str(out)]
                if not original:
                    command += [*acceleration, "--trial-offset", str(offset)]
                run_command(command, out, 120)
                return read_result(out, expected_trials=experiments, expected_generations=11,
                                   trial_offset=0 if original else offset,
                                   require_verification=True)

            try:
                reference = launch("reference", 6, 0, True)
                full = launch("full", 6, 0)
                compare_results(reference, full)
                shards = [launch("shard_0_2", 2, 0), launch("shard_2_1", 1, 2), launch("shard_3_3", 3, 3)]
                counts = [count for shard in shards for count in shard["best_counts"]]
                histories = [row[1:] for shard in shards for row in shard["history"][1:]
                             if row[0].startswith("Trial_")]
                full_histories = [row[1:] for row in full["history"][1:] if row[0].startswith("Trial_")]
                if counts != full["best_counts"] or histories != full_histories:
                    raise AssertionError("concatenated shard counts/histories differ from full run")
                # CLI keeps the earliest trial on an equal minimum; selecting the
                # earliest shard with this minimum preserves the same circuit.
                winning_shard = min(shards, key=lambda result: result["global_minimum"])
                if winning_shard["best_circuit"] != full["best_circuit"]:
                    raise AssertionError("reconstructed global best gate list differs")
                report["passed"] += 1
                report["records"].append({"algorithm": algorithm, "status": "passed"})
                print(f"PASS algorithm {algorithm}: 2 + 1 + 3 trial shards equal the original 6-trial run", flush=True)
            except Exception as exc:
                report["failed"] += 1
                report["records"].append({"algorithm": algorithm, "status": "failed", "error": str(exc)})
                print(f"FAIL algorithm {algorithm}: {exc}", file=sys.stderr, flush=True)
        (root / "shard_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Shard regression: {report['passed']} passed, {report['failed']} failed")
        return 1 if report["failed"] else 0
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
