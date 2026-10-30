// quantum_algos.cpp - port of core/AE_QTS.py, core/QTS.py, core/QEA.py.
//
// All three sample N neighbors from the probability layers, evaluate them, and rotate the
// probabilities; they differ only in the update rule (updateQ):
//   AE-QTS : for t = 0 .. N/2-1, the t-th best pulls the t-th worst with step delta/(t+1)
//   QTS    : best vs worst only, step delta
//   QEA    : every bit rotated toward the global-best individual, step delta
#include "qbcs.h"

#include <climits>

namespace qbcs {

// One probability pair: increase P[b], decrease P[w]; boundary correction keeps [0, 1].
static inline void rotate(std::vector<double>& q, int i, int b, int w, double dt) {
    q[2 * i + b] += dt;
    q[2 * i + w] -= dt;
    if (q[2 * i + w] <= 0.0) {
        q[2 * i + b] = 1.0;
        q[2 * i + w] = 0.0;
    }
}

// best-vs-worst rotation over all layers (used by AE-QTS and QTS)
static void rotate_best_vs_worst(QLayers& Q, const Sample& best, const Sample& worst, double dt) {
    for (int L = 0; L < 4; ++L) {
        std::vector<double>& q = Q.layer(L);
        const std::vector<uint8_t>& b = best.layer(L);
        const std::vector<uint8_t>& w = worst.layer(L);
        const int m = static_cast<int>(b.size());
        for (int i = 0; i < m; ++i)
            if (b[i] != w[i]) rotate(q, i, b[i], w[i], dt);
    }
}

// QEA: rotate toward the global best on every bit
static void rotate_toward(QLayers& Q, const Sample& gb, double dt) {
    for (int L = 0; L < 4; ++L) {
        std::vector<double>& q = Q.layer(L);
        const std::vector<uint8_t>& g = gb.layer(L);
        const int m = static_cast<int>(g.size());
        for (int i = 0; i < m; ++i) rotate(q, i, g[i], 1 - g[i], dt);
    }
}

enum class QKind { AEQTS, QTS, QEA };

static RunResult run_quantum(QKind kind, const Encoding& E, const Params& P, Rng& rng,
                             const std::vector<int>& target) {
    const int N = P.num_neighbors;
    const double delta = kind == QKind::AEQTS        ? P.delta_theta_aeqts
                       : kind == QKind::QTS          ? P.delta_theta_qts
                                                     : P.delta_theta_qea;
    const bool track_gb = kind == QKind::QEA;

    Synthesizer S(E);
    QLayers Q;
    Q.init(E);

    std::vector<Sample> nbrs(N);
    std::vector<int> fit(N), idx;
    Sample best_sample, gb;
    RunResult R;
    R.history.assign(P.max_iterations, 0.0);
    int best = INT_MAX;

    for (int iter = 0; iter < P.max_iterations; ++iter) {
        // Step 1-3: neighborhood generation, synthesis, fitness (gate count)
        for (int i = 0; i < N; ++i) {
            sample_individual(E, Q, rng, nbrs[i]);
        }
        evaluate_many(S, nbrs.data(), nbrs.size(), fit.data(), P, target, R.verify_failures);
        argsort_stable(fit, idx);
        const int lb = idx[0];

        // Global best tracking (strict improvement, as in Python)
        if (fit[lb] < best) {
            best = fit[lb];
            best_sample = nbrs[lb];
            if (track_gb) gb = nbrs[lb];
        }

        // Step 4: probability update
        switch (kind) {
            case QKind::AEQTS: {
                const int limit = N / 2;  // int(num_neighbors / 2)
                for (int t = 0; t < limit; ++t)
                    rotate_best_vs_worst(Q, nbrs[idx[t]], nbrs[idx[N - 1 - t]], delta / (t + 1));
                break;
            }
            case QKind::QTS:
                rotate_best_vs_worst(Q, nbrs[idx[0]], nbrs[idx[N - 1]], delta);
                break;
            case QKind::QEA:
                rotate_toward(Q, gb, delta);
                break;
        }

        R.history[iter] = best;
    }

    R.best_gate_count = best;
    S.synthesize(best_sample, R.best_circuit);
    return R;
}

RunResult run_aeqts(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    return run_quantum(QKind::AEQTS, E, P, rng, target);
}
RunResult run_qts(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    return run_quantum(QKind::QTS, E, P, rng, target);
}
RunResult run_qea(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    return run_quantum(QKind::QEA, E, P, rng, target);
}

}  // namespace qbcs
