"""Dump an AE-QTS / QTS / QEA trace from the ORIGINAL Python implementation for the C++
replay test (tests/replay_test.cpp).

Usage:  python dump_replay.py <path/to/AEQTS_boolean_circuit> <num_bits> <problem_idx> \
                              <iterations> <algo: aeqts|qts|qea> <seed> <out.txt>

For every iteration the trace records the sampled neighbours (flattened in the C++ layer
order), the circuits Python synthesizes for them, the sorted neighbour indices and the
probability layers after Python's updateQ. The C++ side re-synthesizes the same samples and
re-applies its own update rule, then compares gate lists and probabilities exactly.
"""
import sys, random
import numpy as np

root, nbits, pidx, iters, algo, seed, out_path = sys.argv[1:8]
nbits, pidx, iters, seed = int(nbits), int(pidx), int(iters), int(seed)
sys.path.insert(0, root)
from utils.data_loader import DataLoader
from utils.init_state import find_cycles, build_encode, gen_nbrs
from utils.topology import decode_and_synthesize
import core.AE_QTS as AE
import core.QTS as QT
import core.QEA as QE

random.seed(seed); np.random.seed(seed)
tt = DataLoader().get_output(nbits, pidx)
cycles, _ = find_cycles(tt, True)
q1, q2, q3, q4, enc, traj = build_encode(cycles)
N = 10
delta = {'aeqts': 0.01, 'qts': 0.01, 'qea': 0.0005}[algo]
num_cycles = len(cycles)

def flat_l1(a): return [int(x) for row in np.asarray(a).tolist() for x in row]
def flat_l2(a): return [int(x) for row in a for x in row]
def flat_l3(a):
    out = []
    for cyc in a:
        for step in cyc:
            if step[0] == 999: continue
            for node in step: out.extend(int(x) for x in node)
    return out
def flat_l4(a):
    out = []
    for cyc in a:
        for step in cyc:
            if len(step) <= 1: continue
            out.extend(int(x) for x in step)
    return out
def flat_q(q, layer):
    out = []
    if layer == 1:
        for m in q:
            for row in m: out.extend([row[0], row[1]])
    elif layer == 2:
        for m in q:
            for row in m: out.extend([row[0], row[1]])
    elif layer == 3:
        for cyc in q:
            for step in cyc:
                if len(step) == 1: continue   # [[999,999]] marker
                for node in step:
                    for row in node: out.extend([row[0], row[1]])
    else:
        for cyc in q:
            for step in cyc:
                if isinstance(step, list): continue  # [[999,999]] marker
                for row in step: out.extend([row[0], row[1]])
    return out

with open(out_path, 'w') as f:
    f.write(f"HEADER {nbits} {pidx} {iters} {N} {algo} {num_cycles}\n")
    f.write("TT " + " ".join(map(str, tt)) + "\n")
    f.write("Q0 " + " ".join(repr(float(x)) for x in flat_q(q1,1) + flat_q(q2,2) + flat_q(q3,3) + flat_q(q4,4)) + "\n")
    gb = None
    global_best = float('inf')
    for it in range(iters):
        nbr1, nbr2, nbr3, nbr4 = gen_nbrs(q1, q2, q3, q4, N)
        sols = decode_and_synthesize(nbr1, nbr2, nbr3, nbr4, enc, nbits, N, traj)
        f.write(f"ITER {it}\n")
        for i in range(N):
            f.write("L1 " + " ".join(map(str, flat_l1(nbr1[i]))) + "\n")
            f.write("L2 " + " ".join(map(str, flat_l2(nbr2[i]))) + "\n")
            f.write("L3 " + " ".join(map(str, flat_l3(nbr3[i]))) + "\n")
            f.write("L4 " + " ".join(map(str, flat_l4(nbr4[i]))) + "\n")
        for i in range(N):
            gates = sols[i][0]
            f.write(f"CIRCUIT {len(gates)} " + " ".join(str(v) for g in gates for v in g) + "\n")
        metrics = sorted([(s[1], k) for k, s in enumerate(sols)], key=lambda x: x[0])
        sorted_idx = [m[1] for m in metrics]
        f.write("SORTED " + " ".join(map(str, sorted_idx)) + "\n")
        if algo == 'aeqts':
            AE.updateQ(q1, q2, q3, q4, N, nbr1, nbr2, nbr3, nbr4, sorted_idx, num_cycles, delta)
        elif algo == 'qts':
            QT.updateQ(q1, q2, q3, q4, N, nbr1, nbr2, nbr3, nbr4, sorted_idx, num_cycles, delta)
        else:
            lb = int(np.argmin([s[1] for s in sols]))
            if sols[lb][1] < global_best:
                global_best = sols[lb][1]
                gb = (nbr1[lb], nbr2[lb], nbr3[lb], nbr4[lb])
            QE.updateQ(q1, q2, q3, q4, gb[0], gb[1], gb[2], gb[3], num_cycles, delta)
        f.write("Q " + " ".join(repr(float(x)) for x in flat_q(q1,1) + flat_q(q2,2) + flat_q(q3,3) + flat_q(q4,4)) + "\n")
print("dumped", out_path)
