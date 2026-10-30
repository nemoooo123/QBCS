// replay_test.cpp - consumes a trace written by tests/dump_replay.py and checks that
//   (1) the C++ Synthesizer produces exactly the same gate list as Python for every sample,
//   (2) the C++ fitness ordering matches Python's sorted indices,
//   (3) the C++ AE-QTS / QTS / QEA update reproduces Python's probability layers bit-for-bit.
//
// Build: cmake -S . -B build -DQBCS_ENABLE_CUDA=OFF
//        cmake --build build --target replay_test
#include "qbcs.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace qbcs;

// The update rules are static inside quantum_algos.cpp; re-implemented here verbatim so the
// test stays independent of the driver loop.
static void rotate(std::vector<double>& q, int i, int b, int w, double dt) {
    q[2 * i + b] += dt; q[2 * i + w] -= dt;
    if (q[2 * i + w] <= 0.0) { q[2 * i + b] = 1.0; q[2 * i + w] = 0.0; }
}
static void best_vs_worst(QLayers& Q, const Sample& b, const Sample& w, double dt) {
    for (int L = 0; L < 4; ++L) {
        auto& q = Q.layer(L); const auto& sb = b.layer(L); const auto& sw = w.layer(L);
        for (size_t i = 0; i < sb.size(); ++i) if (sb[i] != sw[i]) rotate(q, (int)i, sb[i], sw[i], dt);
    }
}
static void toward(QLayers& Q, const Sample& g, double dt) {
    for (int L = 0; L < 4; ++L) {
        auto& q = Q.layer(L); const auto& sg = g.layer(L);
        for (size_t i = 0; i < sg.size(); ++i) rotate(q, (int)i, sg[i], 1 - sg[i], dt);
    }
}

static std::vector<uint8_t> read_bits(std::istringstream& in) {
    std::vector<uint8_t> v; int x;
    while (in >> x) v.push_back((uint8_t)x);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: replay_test trace.txt\n"; return 2; }
    std::ifstream f(argv[1]);
    if (!f) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }

    std::string line, tag;
    int nbits, pidx, iters, N, num_cycles; std::string algo;
    std::getline(f, line); { std::istringstream in(line); in >> tag >> nbits >> pidx >> iters >> N >> algo >> num_cycles; }
    std::getline(f, line); std::vector<int> tt; { std::istringstream in(line); in >> tag; int v; while (in >> v) tt.push_back(v); }

    auto cycles = find_cycles(tt);
    Encoding E = build_encoding(cycles, nbits);
    if (E.num_cycles != num_cycles) { std::cerr << "cycle count mismatch\n"; return 1; }
    const int qsize = 2 * (E.l1_size + E.l2_size + E.l3_size + E.l4_size);

    auto read_q = [&](const std::string& l, std::vector<double>& out) {
        std::istringstream in(l); in >> tag; out.clear(); std::string s;
        while (in >> s) out.push_back(std::strtod(s.c_str(), nullptr));
    };
    auto flat_q = [&](const QLayers& Q, std::vector<double>& out) {
        out.clear();
        for (int L = 0; L < 4; ++L) out.insert(out.end(), Q.layer(L).begin(), Q.layer(L).end());
    };

    QLayers Q; Q.init(E);
    std::vector<double> q0; std::getline(f, line); read_q(line, q0);
    std::vector<double> mine; flat_q(Q, mine);
    if ((int)q0.size() != qsize || q0 != mine) { std::cerr << "initial Q mismatch: " << q0.size() << " vs " << qsize << "\n"; return 1; }

    const double delta = algo == "qea" ? 0.0005 : 0.01;
    Synthesizer S(E);
    std::vector<Sample> nbrs(N); std::vector<int> fit(N), idx;
    Sample gb; int global_best = 1 << 30;
    long circuits_checked = 0, gates_checked = 0;

    for (int it = 0; it < iters; ++it) {
        std::getline(f, line);
        if (line.rfind("ITER", 0) != 0) { std::cerr << "expected ITER at iteration " << it << "\n"; return 1; }
        for (int i = 0; i < N; ++i) {
            for (int L = 0; L < 4; ++L) {
                std::getline(f, line); std::istringstream in(line); in >> tag;
                nbrs[i].layer(L) = read_bits(in);
                if ((int)nbrs[i].layer(L).size() != E.layer_size(L)) {
                    std::cerr << "iter " << it << " nbr " << i << " layer " << L + 1 << " size " << nbrs[i].layer(L).size()
                              << " != " << E.layer_size(L) << "\n"; return 1;
                }
            }
        }
        Circuit c;
        for (int i = 0; i < N; ++i) {
            std::getline(f, line); std::istringstream in(line); in >> tag; int cnt; in >> cnt;
            fit[i] = S.synthesize(nbrs[i], c);
            if (fit[i] != cnt) { std::cerr << "iter " << it << " nbr " << i << ": gate count " << fit[i] << " != python " << cnt << "\n"; return 1; }
            for (int k = 0; k < cnt; ++k) {
                for (int b = nbits - 1; b >= 0; --b) {
                    int v; in >> v;
                    int expect = (b == c[k].target) ? 3 : (int)((c[k].ctrl >> b) & 1u);
                    if (v != expect) { std::cerr << "iter " << it << " nbr " << i << " gate " << k << " bit " << b << ": " << expect << " != python " << v << "\n"; return 1; }
                }
            }
            if (!verify_circuit(c, nbits, tt)) { std::cerr << "iter " << it << " nbr " << i << ": circuit does not realize the truth table\n"; return 1; }
            ++circuits_checked; gates_checked += cnt;
        }
        std::getline(f, line); std::vector<int> py_sorted; { std::istringstream in(line); in >> tag; int v; while (in >> v) py_sorted.push_back(v); }
        argsort_stable(fit, idx);
        if (idx != py_sorted) { std::cerr << "iter " << it << ": sorted indices differ\n"; return 1; }

        if (algo == "aeqts") { for (int t = 0; t < N / 2; ++t) best_vs_worst(Q, nbrs[idx[t]], nbrs[idx[N - 1 - t]], delta / (t + 1)); }
        else if (algo == "qts") { best_vs_worst(Q, nbrs[idx[0]], nbrs[idx[N - 1]], delta); }
        else { if (fit[idx[0]] < global_best) { global_best = fit[idx[0]]; gb = nbrs[idx[0]]; } toward(Q, gb, delta); }

        std::getline(f, line); std::vector<double> pyq; read_q(line, pyq);
        flat_q(Q, mine);
        if (pyq != mine) {
            for (size_t k = 0; k < mine.size(); ++k)
                if (pyq[k] != mine[k]) { std::cerr << "iter " << it << ": Q[" << k << "] " << mine[k] << " != python " << pyq[k] << "\n"; break; }
            return 1;
        }
    }
    std::cout << "REPLAY OK: " << algo << " n=" << nbits << " idx=" << pidx << " iters=" << iters
              << " | circuits identical: " << circuits_checked << " (" << gates_checked << " gates), all realize the truth table"
              << " | probability layers identical after every update\n";
    return 0;
}
