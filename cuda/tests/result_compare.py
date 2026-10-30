"""Read and compare qbcs CLI results without comparing timing columns.

Used by regression.py and tools/benchmark.py.  Standard library only.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def run_command(command: list[str], output_dir: Path, timeout: float) -> dict[str, Any]:
    """Run an explicit argv, preserving logs on success and failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    start = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        (output_dir / "timeout.txt").write_text(str(exc), encoding="utf-8")
        raise RuntimeError(f"timed out after {timeout}s: {output_dir}") from exc
    elapsed = time.perf_counter() - start
    (output_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"exit {process.returncode}: {' '.join(command)}\n"
            f"{process.stderr[-4000:]}\n{process.stdout[-2000:]}"
        )
    return {"wall_seconds": elapsed, "stdout": process.stdout,
            "stderr": process.stderr, "command": command}


def read_result(output_dir: Path, *, expected_trials: int | None = None,
                expected_generations: int | None = None,
                trial_offset: int = 0,
                require_verification: bool = False) -> dict[str, Any]:
    """Return every history value plus the published best circuit and summary.

    Per-trial circuits are not present in the original CLI's output.  The checked
    circuit is therefore the global best circuit, including its complete order.
    Every trial's complete convergence curve and final gate count are checked.
    """
    csv_paths = list(output_dir.glob("*_bit/*_Results/*.csv"))
    txt_paths = list(output_dir.glob("*_bit/*_Results/*.txt"))
    if len(csv_paths) != 1 or len(txt_paths) != 1:
        raise AssertionError(f"expected one CSV and one TXT below {output_dir}, "
                             f"found {len(csv_paths)} and {len(txt_paths)}")
    with csv_paths[0].open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows or "Execution_Time(s)" not in rows[0]:
        raise AssertionError(f"missing execution-time column in {csv_paths[0]}")
    time_col = rows[0].index("Execution_Time(s)")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise AssertionError(f"malformed CSV row width in {csv_paths[0]}")
    history = [row[:time_col] + row[time_col + 1:] for row in rows]
    trial_rows = [row for row in history[1:] if row[0].startswith("Trial_")]
    if expected_trials is not None and len(trial_rows) != expected_trials:
        raise AssertionError(f"trial row count {len(trial_rows)} != {expected_trials}")
    if expected_generations is not None:
        wanted = [""] + [f"Gen_{i + 1}" for i in range(expected_generations)]
        if history[0] != wanted:
            raise AssertionError("generation header differs from requested iterations")
    expected_names = ([f"Trial_{trial_offset + i + 1}" for i in range(len(trial_rows))]
                      + ["Average_Convergence", "Std_Deviation"])
    if [row[0] for row in history[1:]] != expected_names:
        raise AssertionError("missing, repeated, or reordered history rows")

    summary = txt_paths[0].read_text(encoding="utf-8-sig")

    def field(pattern: str) -> str:
        match = re.search(pattern, summary, flags=re.MULTILINE)
        if not match:
            raise AssertionError(f"summary missing {pattern!r}: {txt_paths[0]}")
        return match.group(1).strip()

    counts = ast.literal_eval(field(r"^Best Gate Counts per Trial:\s*(\[.*\])$"))
    minimum = int(field(r"^Global Minimum Gate Count:\s*(\d+)\s*$"))
    final = field(r"^Final Result \(Gen \d+\):\s*(.+)$")
    circuit = ast.literal_eval(field(r"^Best Circuit Structure:\s*\n(\[.*\])$"))
    verification_match = re.search(r"^Circuit Logic Verification Failures:\s*(\d+)\s*$",
                                   summary, flags=re.MULTILINE)
    failures = int(verification_match.group(1)) if verification_match else None
    if require_verification and failures is None:
        raise AssertionError("--verify was requested but the summary has no verification result")
    if failures not in (None, 0):
        raise AssertionError(f"circuit verification reported {failures} failures")
    if len(counts) != len(trial_rows) or minimum != min(counts):
        raise AssertionError("inconsistent per-trial best counts or global minimum")
    if len(circuit) != minimum:
        raise AssertionError(f"best circuit length {len(circuit)} != gate count {minimum}")
    for count, row in zip(counts, trial_rows):
        if len(row) > 1 and float(row[-1]) != count:
            raise AssertionError("final history value differs from trial's best gate count")
    for gate in circuit:
        if not isinstance(gate, list) or gate.count(3) != 1 or any(v not in (0, 1, 3) for v in gate):
            raise AssertionError(f"malformed full-control Toffoli gate: {gate}")
    return {"history": history, "best_counts": counts, "global_minimum": minimum,
            "final_result": final, "best_circuit": circuit, "verification_failures": failures}


def compare_results(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Require exact output equality, except absent versus zero optional verification."""
    for key in ("best_counts", "global_minimum", "final_result", "best_circuit"):
        if reference[key] != candidate[key]:
            expected = repr(reference[key])[:1500]
            actual = repr(candidate[key])[:1500]
            raise AssertionError(f"{key} differs:\nreference: {expected}\ncandidate: {actual}")
    left, right = reference["history"], candidate["history"]
    if len(left) != len(right):
        raise AssertionError(f"history row counts differ: {len(left)} != {len(right)}")
    for row, (a, b) in enumerate(zip(left, right)):
        if len(a) != len(b):
            raise AssertionError(f"history row {row} column counts differ")
        for column, (x, y) in enumerate(zip(a, b)):
            if x != y:
                raise AssertionError(f"history {a[0] or 'header'}, column {column}: {x!r} != {y!r}")


def result_digest(result: dict[str, Any]) -> str:
    deterministic = {k: v for k, v in result.items() if k != "verification_failures"}
    return hashlib.sha256(json.dumps(deterministic, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
