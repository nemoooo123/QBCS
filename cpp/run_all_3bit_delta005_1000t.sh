#!/usr/bin/env bash
# Run AE-QTS (algo=1), N=50, delta=0.005, over all 40320 3-bit permutation problems.
# 1000 trials x 1000 generations each, full CPU (all cores).
# Then computes the 0~13 gate-count distribution.
set -u

cd "$(dirname "$0")"

LOCK=run_all_3bit_delta005_1000t.lock
if [[ -f "$LOCK" ]]; then
    echo "Lock file $LOCK exists — another instance may be running. Aborting." >&2
    exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

EXE=./bin/qbcs.exe
DATA=data/problems_3bit_all.txt
LOG=run_all_3bit_delta005_1000t.log
TOTAL=40320
N=50
DELTA=0.005
OUT=exp_3bit_delta005_1000t
RESULT_DIR="$OUT/3_bit/AE-QTS_Results"
mkdir -p "$RESULT_DIR"

echo "$(date '+%F %T') START AE-QTS delta=$DELTA N=$N experiments=1000 total=$TOTAL" >> "$LOG"

for ((idx=1; idx<=TOTAL; idx++)); do
    out_txt="$RESULT_DIR/AE-QTS_3_${idx}.txt"
    if [[ -f "$out_txt" ]]; then
        continue
    fi
    "$EXE" 3 "$idx" 1 \
        --experiments 1000 \
        --iterations 1000 \
        --neighbors "$N" \
        --delta "$DELTA" \
        --data "$DATA" \
        --out "$OUT" \
        >> "$LOG" 2>&1

    if (( idx % 500 == 0 )); then
        echo "$(date '+%F %T') progress: $idx / $TOTAL done" >> "$LOG"
    fi
done

echo "$(date '+%F %T') SWEEP DONE ($TOTAL problems)" >> "$LOG"

DIST_FILE="gatecount_dist_delta005_1000t.txt"
grep -h "Global Minimum Gate Count" "$RESULT_DIR"/*.txt \
    | sed -E 's/.*Gate Count: *([0-9]+).*/\1/' \
    | sort -n | uniq -c > "$DIST_FILE"

echo "$(date '+%F %T') distribution written to $DIST_FILE" >> "$LOG"
echo "$(date '+%F %T') ALL DONE" >> "$LOG"
