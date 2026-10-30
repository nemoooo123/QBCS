#!/usr/bin/env python3
"""Run the original Python sample/update replay and check the C++ replay binary.

This checks sampling outputs supplied by Python, complete synthesized gate
lists, fitness order, and the replay's probability updates for algorithms 1..3.
It does not claim that Python and C++ use an identical RNG or that the replay's
copied update helper alone validates the production driver. regression.py
separately compares the actual executable across all 10 algorithms.

Existing traces may be replayed without importing any Python project modules:
  python tests/python_replay.py --replay-binary build/replay_test --traces trace_*.txt

To generate fresh traces, provide the unmodified original Python project:
  python tests/python_replay.py --replay-binary build/replay_test --python-root ../original_python
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command: list[str]) -> str:
    process = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if process.returncode:
        raise RuntimeError(f"exit {process.returncode}: {' '.join(command)}\n{process.stdout}\n{process.stderr}")
    return process.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--python-root", type=Path)
    parser.add_argument("--traces", type=Path, nargs="*")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, help="retain generated traces and JSON report here")
    args = parser.parse_args()
    binary = args.replay_binary.resolve()
    if not binary.is_file():
        parser.error(f"replay binary does not exist: {binary}")
    if not args.traces and args.python_root is None:
        parser.error("provide --traces or --python-root")
    temporary = tempfile.TemporaryDirectory(prefix="qbcs-python-replay-") if args.output is None else None
    out = Path(temporary.name) if temporary else args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    try:
        traces = [trace.resolve() for trace in (args.traces or [])]
        if args.python_root is not None:
            root = args.python_root.resolve()
            if not (root / "utils/topology.py").is_file():
                parser.error(f"original Python project not found at {root}")
            # numpy and pandas are imported by the original modules.  Keep
            # dependency failures explicit instead of editing the reference.
            for bits, algorithm, iterations in [(3, "aeqts", 20), (5, "qts", 12),
                                                  (5, "qea", 12), (8, "aeqts", 3)]:
                trace = out / f"python_{bits}_{algorithm}.txt"
                message = run([args.python, str(Path(__file__).with_name("dump_replay.py")),
                               str(root), str(bits), "1", str(iterations), algorithm, "7", str(trace)])
                print(message, flush=True)
                traces.append(trace)
        for trace in traces:
            message = run([str(binary), str(trace)])
            print(message, flush=True)
            records.append({"trace": str(trace), "status": "passed", "output": message})
        (out / "python_replay_report.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"Python replay: {len(records)} traces passed.")
        return 0
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
