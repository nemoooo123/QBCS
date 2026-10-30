#!/usr/bin/env bash
# Run the bit-exact replay test: Python samples -> C++ synthesis/update -> compare.
#   tests/run_replay.sh <path/to/AEQTS_boolean_circuit (Python project)> [num_bits problem_idx iterations algo seed]
set -euo pipefail
PY_ROOT=${1:?path to the Python project (folder containing main.py)}
NB=${2:-5}; IDX=${3:-1}; IT=${4:-200}; ALGO=${5:-aeqts}; SEED=${6:-7}
HERE=$(cd "$(dirname "$0")/.." && pwd)
TRACE=$(mktemp /tmp/qbcs_trace_XXXX.txt)
trap 'rm -f "$TRACE"' EXIT
REPLAY_BINARY=${QBCS_REPLAY_BINARY:-$HERE/build-replay/replay_test}

if [ ! -x "$REPLAY_BINARY" ]; then
  cmake -S "$HERE" -B "$HERE/build-replay" -DQBCS_ENABLE_CUDA=OFF -DQBCS_BUILD_TESTS=ON
  cmake --build "$HERE/build-replay" --target replay_test --parallel
fi
python3 "$HERE/tests/dump_replay.py" "$PY_ROOT" "$NB" "$IDX" "$IT" "$ALGO" "$SEED" "$TRACE"
"$REPLAY_BINARY" "$TRACE"
