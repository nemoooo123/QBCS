#!/usr/bin/env bash
# Run the bit-exact replay test: Python samples -> C++ synthesis/update -> compare.
#   tests/run_replay.sh <path/to/AEQTS_boolean_circuit (Python project)> [num_bits problem_idx iterations algo seed]
set -euo pipefail
PY_ROOT=${1:?path to the Python project (folder containing main.py)}
NB=${2:-5}; IDX=${3:-1}; IT=${4:-200}; ALGO=${5:-aeqts}; SEED=${6:-7}
HERE=$(cd "$(dirname "$0")/.." && pwd)
TRACE=$(mktemp /tmp/qbcs_trace_XXXX.txt)

if [ ! -x "$HERE/replay_test" ]; then
  g++ -std=c++17 -O2 -I "$HERE/src" "$HERE/tests/replay_test.cpp" "$HERE"/src/encoding.cpp "$HERE"/src/synth.cpp \
      "$HERE"/src/algorithms_common.cpp "$HERE"/src/quantum_algos.cpp "$HERE"/src/baseline_algos.cpp -o "$HERE/replay_test"
fi
python3 "$HERE/tests/dump_replay.py" "$PY_ROOT" "$NB" "$IDX" "$IT" "$ALGO" "$SEED" "$TRACE"
"$HERE/replay_test" "$TRACE"
rm -f "$TRACE"
