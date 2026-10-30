#!/usr/bin/env bash
# Run GA (pc=0.9, pm=0.001) and TS (tabu=150) over all 40320 3-bit permutation problems.
# N=50, 1000 trials x 1000 generations each, full CPU (all cores).
set -u

cd "$(dirname "$0")"

LOCK=run_all_3bit_ga_ts.lock
if [[ -f "$LOCK" ]]; then
    echo "Lock file $LOCK exists — another instance may be running. Aborting." >&2
    exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

EXE=./bin/qbcs.exe
DATA=data/problems_3bit_all.txt
LOG=run_all_3bit_ga_ts.log
TOTAL=40320
N=50
OUT=exp_3bit_ga_ts

echo "$(date '+%F %T') START GA(pc=0.9,pm=0.001) + TS(tabu=150), N=$N, total=$TOTAL" >> "$LOG"

# --- GA ---
RESULT_DIR="$OUT/3_bit/GA_Results"
mkdir -p "$RESULT_DIR"
echo "$(date '+%F %T') === GA START ===" >> "$LOG"
for ((idx=1; idx<=TOTAL; idx++)); do
    out_txt="$RESULT_DIR/GA_3_${idx}.txt"
    if [[ -f "$out_txt" ]]; then continue; fi
    "$EXE" 3 "$idx" 4 \
        --experiments 1000 --iterations 1000 --neighbors "$N" \
        --ga-pc 0.9 --ga-pm 0.001 \
        --data "$DATA" --out "$OUT" \
        >> "$LOG" 2>&1
    if (( idx % 1000 == 0 )); then
        echo "$(date '+%F %T') GA progress: $idx / $TOTAL done" >> "$LOG"
    fi
done
echo "$(date '+%F %T') === GA DONE ===" >> "$LOG"

# --- TS ---
RESULT_DIR="$OUT/3_bit/TS_Results"
mkdir -p "$RESULT_DIR"
echo "$(date '+%F %T') === TS START ===" >> "$LOG"
for ((idx=1; idx<=TOTAL; idx++)); do
    out_txt="$RESULT_DIR/TS_3_${idx}.txt"
    if [[ -f "$out_txt" ]]; then continue; fi
    "$EXE" 3 "$idx" 6 \
        --experiments 1000 --iterations 1000 --neighbors "$N" \
        --tabu 150 \
        --data "$DATA" --out "$OUT" \
        >> "$LOG" 2>&1
    if (( idx % 1000 == 0 )); then
        echo "$(date '+%F %T') TS progress: $idx / $TOTAL done" >> "$LOG"
    fi
done
echo "$(date '+%F %T') === TS DONE ===" >> "$LOG"

echo "$(date '+%F %T') ALL DONE" >> "$LOG"
