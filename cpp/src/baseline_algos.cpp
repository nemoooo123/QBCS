// baseline_algos.cpp - port of core/GA.py, DE.py, TS.py, PSO.py, WOA.py, ABC.py.
//
// All operate on the same flat 4-layer Sample. Wherever Python touched the 999 markers
// (mutating the [0] placeholder of a Hamming-distance-1 step in Layer 4, "FIX" tabu moves)
// the effect on the synthesized circuit was nil; those positions simply do not exist here,
// and the random choices that could land on them are reproduced as no-ops so the move
// distributions stay the same.
#include "qbcs.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <deque>

namespace qbcs {

namespace {

// Flip every bit of a layer with probability `rate`.
inline void mutate_layer(std::vector<uint8_t>& bits, double rate, Rng& rng) {
    for (auto& b : bits)
        if (rng.uniform() < rate) b = 1 - b;
}

// Mutate all four layers; Layer 4 is repaired step by step afterwards.
inline void mutate_all(const Encoding& E, Sample& s, double rate, Rng& rng) {
    mutate_layer(s.l1, rate, rng);
    mutate_layer(s.l2, rate, rng);
    mutate_layer(s.l3, rate, rng);
    for (int g = 0; g < E.total_steps; ++g) {
        const int a = E.l4_off[g], b = E.l4_off[g + 1];
        for (int i = a; i < b; ++i)
            if (rng.uniform() < rate) s.l4[i] = 1 - s.l4[i];
        repair_l4_step_at(E, s.l4, g, rng);
    }
}

// random.choice([idx for idx in range(n) if idx != i])
inline int random_other(Rng& rng, int n, int i) {
    int j = rng.below(n - 1);
    return j >= i ? j + 1 : j;
}

// np.argmin: first index of the minimum
inline int argmin(const std::vector<int>& v) {
    int b = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] < v[b]) b = i;
    return b;
}

}  // namespace

// =========================================================================================
// GA - Genetic Algorithm (tournament selection, single-point crossover per layer at cycle
// granularity, bit-flip mutation, elitist (N+1 -> N) truncation)
// =========================================================================================
RunResult run_ga(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    const int N = P.num_neighbors;
    Synthesizer S(E);
    QLayers Q;
    Q.init(E);
    RunResult R;
    R.history.assign(P.max_iterations, 0.0);

    std::vector<Sample> pop(N), children(N + 1);
    std::vector<int> fit(N), cfit(N + 1), idx, tour;
    for (int i = 0; i < N; ++i) {
        sample_individual(E, Q, rng, pop[i]);
        fit[i] = evaluate(S, pop[i], P, target, R.verify_failures);
    }
    int best = fit[argmin(fit)];
    Sample best_sample = pop[argmin(fit)];

    const int K = P.ga_tournament_size();
    auto tournament = [&]() {
        rng.sample_indices(N, K, tour);
        int b = tour[0];
        for (int j = 1; j < K; ++j)
            if (fit[tour[j]] < fit[b]) b = tour[j];  // min(): first minimal element
        return b;
    };

    for (int iter = 0; iter < P.max_iterations; ++iter) {
        const int elite_idx = argmin(fit);

        for (int c = 0; c < N; ++c) {
            const int p1 = tournament();
            const int p2 = tournament();
            Sample& child = children[c];
            if (rng.uniform() < P.ga_pc) {
                for (int L = 0; L < 4; ++L)
                    crossover_layer(E, L, pop[p1], pop[p2], rng.randint(0, E.num_cycles), child);
            } else {
                child = pop[p1];
            }
            mutate_all(E, child, P.ga_pm, rng);
            cfit[c] = evaluate(S, child, P, target, R.verify_failures);
        }
        // merge the previous elite (fitness already known) and keep the best N
        children[N] = pop[elite_idx];
        cfit[N] = fit[elite_idx];
        argsort_stable(cfit, idx);
        std::vector<Sample> next(N);
        for (int i = 0; i < N; ++i) { next[i] = children[idx[i]]; fit[i] = cfit[idx[i]]; }
        pop.swap(next);

        const int lb = argmin(fit);
        if (fit[lb] < best) { best = fit[lb]; best_sample = pop[lb]; }
        R.history[iter] = best;
    }
    R.best_gate_count = best;
    S.synthesize(best_sample, R.best_circuit);
    return R;
}

// =========================================================================================
// DE - Differential Evolution (binary: mutant = r1 | (F & (r2 ^ r3)), F in {0,1};
// binomial crossover with CR; greedy replacement if trial <= target)
// =========================================================================================
RunResult run_de(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    const int N = P.num_neighbors;
    Synthesizer S(E);
    QLayers Q;
    Q.init(E);
    RunResult R;
    R.history.assign(P.max_iterations, 0.0);

    std::vector<Sample> pop(N), next(N);
    std::vector<int> fit(N), nfit(N), cand;
    for (int i = 0; i < N; ++i) {
        sample_individual(E, Q, rng, pop[i]);
        fit[i] = evaluate(S, pop[i], P, target, R.verify_failures);
    }
    int best = INT_MAX;
    Sample best_sample;
    Sample trial;
    trial.resize(E);

    for (int iter = 0; iter < P.max_iterations; ++iter) {
        for (int i = 0; i < N; ++i) {
            const int F = rng.below(2);
            // three distinct individuals different from i
            rng.sample_indices(N - 1, 3, cand);
            for (int& c : cand) if (c >= i) ++c;
            const Sample& r1 = pop[cand[0]];
            const Sample& r2 = pop[cand[1]];
            const Sample& r3 = pop[cand[2]];
            const Sample& x = pop[i];

            for (int L = 0; L < 3; ++L) {
                const std::vector<uint8_t>& a = r1.layer(L);
                const std::vector<uint8_t>& b = r2.layer(L);
                const std::vector<uint8_t>& c = r3.layer(L);
                const std::vector<uint8_t>& xi = x.layer(L);
                std::vector<uint8_t>& t = trial.layer(L);
                for (size_t k = 0; k < a.size(); ++k) {
                    const uint8_t mutant = static_cast<uint8_t>(a[k] | (F & (b[k] ^ c[k])));
                    t[k] = (rng.uniform() <= P.de_cr) ? mutant : xi[k];
                }
            }
            for (int g = 0; g < E.total_steps; ++g) {
                for (int k = E.l4_off[g]; k < E.l4_off[g + 1]; ++k) {
                    const uint8_t mutant = static_cast<uint8_t>(r1.l4[k] | (F & (r2.l4[k] ^ r3.l4[k])));
                    trial.l4[k] = (rng.uniform() <= P.de_cr) ? mutant : x.l4[k];
                }
                repair_l4_step_at(E, trial.l4, g, rng);
            }

            const int tfit = evaluate(S, trial, P, target, R.verify_failures);
            if (tfit <= fit[i]) { next[i] = trial; nfit[i] = tfit; }
            else                { next[i] = pop[i]; nfit[i] = fit[i]; }
        }
        pop.swap(next);
        fit.swap(nfit);

        const int lb = argmin(fit);
        if (fit[lb] < best) { best = fit[lb]; best_sample = pop[lb]; }
        R.history[iter] = best;
    }
    R.best_gate_count = best;
    S.synthesize(best_sample, R.best_circuit);
    return R;
}

// =========================================================================================
// TS - Tabu Search with compound moves (one bit flip per layer) and aspiration criterion
// =========================================================================================
namespace {
struct TsMove {
    int cy1, bi1, cy2, bi2, cy3, st3, no3, bi3 /* -1 == "FIX" */, cy4, st4, bi4;
    bool operator==(const TsMove& o) const {
        return cy1 == o.cy1 && bi1 == o.bi1 && cy2 == o.cy2 && bi2 == o.bi2 && cy3 == o.cy3 &&
               st3 == o.st3 && no3 == o.no3 && bi3 == o.bi3 && cy4 == o.cy4 && st4 == o.st4 && bi4 == o.bi4;
    }
};
}  // namespace

RunResult run_ts(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    const int N = P.num_neighbors;
    Synthesizer S(E);
    QLayers Q;
    Q.init(E);
    RunResult R;
    R.history.assign(P.max_iterations, 0.0);

    Sample cur;
    sample_individual(E, Q, rng, cur);
    int best_fit = evaluate(S, cur, P, target, R.verify_failures);
    Sample best_sample = cur;

    std::deque<TsMove> tabu;
    std::vector<Sample> cands(N);
    std::vector<TsMove> moves(N);
    std::vector<int> fit(N), idx;

    for (int iter = 0; iter < P.max_iterations; ++iter) {
        for (int k = 0; k < N; ++k) {
            Sample& nc = cands[k];
            nc = cur;
            TsMove m{};
            // Layer 1
            m.cy1 = rng.below(E.num_cycles);
            m.bi1 = rng.below(E.l1_bits);
            nc.l1[E.l1_cyc_off[m.cy1] + m.bi1] ^= 1;
            // Layer 2
            m.cy2 = rng.below(E.num_cycles);
            m.bi2 = rng.below(E.l2_bits[m.cy2]);
            nc.l2[E.l2_off[m.cy2] + m.bi2] ^= 1;
            // Layer 3: [cycle][step][node][bit]; a marker step (single "999" node) is a FIX move
            m.cy3 = rng.below(E.num_cycles);
            m.st3 = rng.below(E.nsteps[m.cy3]);
            {
                const int g = E.step_off[m.cy3] + m.st3;
                const int d = E.step_d[g], nb = E.step_nb[g];
                m.no3 = rng.below(d);  // len(nc3[cy][st]) == d nodes (1 for a marker step)
                if (nb == 0) {
                    m.bi3 = -1;
                } else {
                    m.bi3 = rng.below(nb);
                    nc.l3[E.l3_off[g] + m.no3 * nb + m.bi3] ^= 1;
                }
            }
            // Layer 4: marker steps have Python length 1 -> bit 0 flipped (no circuit effect)
            m.cy4 = rng.below(E.num_cycles);
            m.st4 = rng.below(E.nsteps[m.cy4]);
            {
                const int g = E.step_off[m.cy4] + m.st4;
                const int len = E.l4_off[g + 1] - E.l4_off[g];
                if (len == 0) {
                    m.bi4 = 0;
                } else {
                    m.bi4 = rng.below(len);
                    nc.l4[E.l4_off[g] + m.bi4] ^= 1;
                    repair_l4_step_at(E, nc.l4, g, rng);
                }
            }
            moves[k] = m;
            fit[k] = evaluate(S, nc, P, target, R.verify_failures);
        }
        argsort_stable(fit, idx);

        bool found = false;
        for (int r = 0; r < N; ++r) {
            const int k = idx[r];
            const bool aspiring = fit[k] < best_fit;
            const bool is_tabu = std::find(tabu.begin(), tabu.end(), moves[k]) != tabu.end();
            if (!is_tabu || aspiring) {
                cur = cands[k];
                if (fit[k] < best_fit) { best_fit = fit[k]; best_sample = cands[k]; }
                tabu.push_back(moves[k]);
                if (static_cast<int>(tabu.size()) > P.ts_tabu_size) tabu.pop_front();
                found = true;
                break;
            }
        }
        if (!found) {  // all moves tabu and none aspiring: take the best candidate anyway
            const int k = idx[0];
            cur = cands[k];
            tabu.push_back(moves[k]);
            if (static_cast<int>(tabu.size()) > P.ts_tabu_size) tabu.pop_front();
        }
        R.history[iter] = best_fit;
    }
    R.best_gate_count = best_fit;
    S.synthesize(best_sample, R.best_circuit);
    return R;
}

// =========================================================================================
// PSO - binary PSO: continuous position/velocity per bit, discretised by x > 0
// =========================================================================================
namespace {
struct Continuous {
    std::vector<double> l[4];
    void resize(const Encoding& E) {
        for (int L = 0; L < 4; ++L) l[L].assign(E.layer_size(L), 0.0);
    }
};
}  // namespace

RunResult run_pso(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    const int N = P.num_neighbors;
    Synthesizer S(E);
    RunResult R;
    R.history.assign(P.max_iterations, 0.0);

    std::vector<Continuous> pos(N), vel(N), pbest_pos(N);
    std::vector<Sample> disc(N);
    std::vector<int> fit(N), pbest_fit(N);

    // gen_pso_init_states(): position in [-6, 6], velocity in [-1, 1], bit = pos > 0
    for (int i = 0; i < N; ++i) {
        pos[i].resize(E);
        vel[i].resize(E);
        disc[i].resize(E);
        for (int L = 0; L < 4; ++L) {
            std::vector<uint8_t>& d = disc[i].layer(L);
            for (size_t k = 0; k < d.size(); ++k) {
                pos[i].l[L][k] = rng.uniform(-6.0, 6.0);
                vel[i].l[L][k] = rng.uniform(-1.0, 1.0);
                d[k] = pos[i].l[L][k] > 0.0 ? 1 : 0;
            }
        }
        repair_l4_all(E, disc[i].l4, rng);
        pbest_pos[i] = pos[i];
        fit[i] = evaluate(S, disc[i], P, target, R.verify_failures);
        pbest_fit[i] = fit[i];
    }
    int gidx = argmin(pbest_fit);
    int gbest = pbest_fit[gidx];
    Continuous gbest_pos = pbest_pos[gidx];
    Sample best_sample = disc[gidx];

    for (int iter = 0; iter < P.max_iterations; ++iter) {
        for (int i = 0; i < N; ++i) {
            for (int L = 0; L < 4; ++L) {
                std::vector<double>& x = pos[i].l[L];
                std::vector<double>& v = vel[i].l[L];
                const std::vector<double>& pb = pbest_pos[i].l[L];
                const std::vector<double>& gb = gbest_pos.l[L];
                std::vector<uint8_t>& d = disc[i].layer(L);
                for (size_t k = 0; k < x.size(); ++k) {
                    const double r1 = rng.uniform(), r2 = rng.uniform();
                    double nv = P.pso_w * v[k] + P.pso_c1 * r1 * (pb[k] - x[k]) + P.pso_c2 * r2 * (gb[k] - x[k]);
                    nv = std::max(std::min(nv, 6.0), -6.0);
                    v[k] = nv;
                    x[k] += nv;
                    d[k] = x[k] > 0.0 ? 1 : 0;
                }
            }
            repair_l4_all(E, disc[i].l4, rng);
        }
        for (int i = 0; i < N; ++i) fit[i] = evaluate(S, disc[i], P, target, R.verify_failures);

        const int lb = argmin(fit);
        for (int i = 0; i < N; ++i)
            if (fit[i] < pbest_fit[i]) { pbest_fit[i] = fit[i]; pbest_pos[i] = pos[i]; }
        if (fit[lb] < gbest) { gbest = fit[lb]; gbest_pos = pos[lb]; best_sample = disc[lb]; }
        R.history[iter] = gbest;
    }
    R.best_gate_count = gbest;
    S.synthesize(best_sample, R.best_circuit);
    return R;
}

// =========================================================================================
// WOA - Whale Optimization Algorithm (encircling / random search = layer crossover toward
// a target, bubble-net spiral = mutation around the leader)
// =========================================================================================
namespace {

void woa_update(const Encoding& E, const Sample& p1, const Sample& tgt, double C, Rng& rng, Sample& child) {
    child = p1;
    const int num_layers = C < 0.6 ? 1 : (C < 1.4 ? 2 : 3);
    std::vector<int> layers;
    rng.sample_indices(4, num_layers, layers);
    for (int L : layers) {
        crossover_layer(E, L, p1, tgt, rng.randint(0, E.num_cycles), child);
        // Layer 4 repair after a cycle-granularity crossover is a no-op: every step comes
        // whole from one parent and both parents are balanced.
    }
}

void woa_spiral(const Encoding& E, const Sample& leader, double b, Rng& rng, Sample& child) {
    child = leader;
    mutate_all(E, child, 0.05 * b, rng);
}

}  // namespace

RunResult run_woa(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    const int N = P.num_neighbors;
    Synthesizer S(E);
    QLayers Q;
    Q.init(E);
    RunResult R;
    R.history.assign(P.max_iterations, 0.0);

    std::vector<Sample> pop(N);
    std::vector<int> fit(N);
    for (int i = 0; i < N; ++i) {
        sample_individual(E, Q, rng, pop[i]);
        fit[i] = evaluate(S, pop[i], P, target, R.verify_failures);
    }
    int bi = argmin(fit);
    int best = fit[bi];
    Sample leader = pop[bi];
    Sample child;

    for (int iter = 1; iter <= P.max_iterations; ++iter) {
        const double progress = static_cast<double>(iter) / P.max_iterations;
        const double a = P.woa_a_end + (P.woa_a_start - P.woa_a_end) * std::pow(1.0 - progress, P.woa_a_decay_power);
        for (int i = 0; i < N; ++i) {
            const double p = rng.uniform();
            const double r = rng.uniform();
            const double A = 2.0 * a * r - a;
            const double C = 2.0 * r;
            if (p < 0.5) {
                if (std::fabs(A) < 1.0) woa_update(E, pop[i], leader, C, rng, child);
                else                    woa_update(E, pop[i], pop[random_other(rng, N, i)], C, rng, child);
            } else {
                woa_spiral(E, leader, P.woa_b, rng, child);
            }
            const int cf = evaluate(S, child, P, target, R.verify_failures);
            if (cf < fit[i]) {
                pop[i] = child;
                fit[i] = cf;
                if (cf < best) { best = cf; leader = child; }
            }
        }
        R.history[iter - 1] = best;
    }
    R.best_gate_count = best;
    S.synthesize(leader, R.best_circuit);
    return R;
}

// =========================================================================================
// ABC - Artificial Bee Colony (employed / onlooker / scout phases; the neighbor operator is
// a single-layer cycle-granularity crossover with a random partner)
// =========================================================================================
namespace {

void abc_neighbor(const Encoding& E, const Sample& p1, const Sample& p2, Rng& rng, Sample& child) {
    child = p1;
    const int L = rng.randint(0, 3);
    crossover_layer(E, L, p1, p2, rng.randint(0, E.num_cycles), child);
    // Layer-4 repair here is a no-op (see woa_update)
}

}  // namespace

RunResult run_abc(const Encoding& E, const Params& P, Rng& rng, const std::vector<int>& target) {
    const int N = P.num_neighbors;
    Synthesizer S(E);
    QLayers Q;
    Q.init(E);
    RunResult R;
    R.history.assign(P.max_iterations, 0.0);

    std::vector<Sample> pop(N);
    std::vector<int> fit(N), trials(N, 0);
    for (int i = 0; i < N; ++i) {
        sample_individual(E, Q, rng, pop[i]);
        fit[i] = evaluate(S, pop[i], P, target, R.verify_failures);
    }
    int bi = argmin(fit);
    int best = fit[bi];
    Sample best_sample = pop[bi];
    Sample child;
    std::vector<double> probs(N);

    auto greedy = [&](int i) {
        const int cf = evaluate(S, child, P, target, R.verify_failures);
        if (cf < fit[i]) { pop[i] = child; fit[i] = cf; trials[i] = 0; }
        else             { ++trials[i]; }
    };

    for (int iter = 0; iter < P.max_iterations; ++iter) {
        // Scout phase first; its evaluations are deducted from this iteration's budget
        int scouts = 0;
        for (int i = 0; i < N; ++i) {
            if (trials[i] > P.abc_limit) {
                ++scouts;
                mutate_all(E, pop[i], 0.3, rng);
                fit[i] = evaluate(S, pop[i], P, target, R.verify_failures);
                trials[i] = 0;
            }
        }
        const int remaining = N - scouts;
        const int emp_n = remaining / 2;
        const int onl_n = remaining - emp_n;

        // Employed bees (food sources 0 .. emp_n-1, as in the Python loop)
        for (int i = 0; i < emp_n; ++i) {
            abc_neighbor(E, pop[i], pop[random_other(rng, N, i)], rng, child);
            greedy(i);
        }
        // Onlooker bees: roulette wheel on 1 / (fitness + 1e-6)
        double total = 0.0;
        for (int i = 0; i < N; ++i) { probs[i] = 1.0 / (fit[i] + 1e-6); total += probs[i]; }
        for (int k = 0; k < onl_n; ++k) {
            const double u = rng.uniform() * total;
            double acc = 0.0;
            int i = N - 1;
            for (int j = 0; j < N; ++j) { acc += probs[j]; if (u < acc) { i = j; break; } }
            abc_neighbor(E, pop[i], pop[random_other(rng, N, i)], rng, child);
            greedy(i);
        }

        const int lb = argmin(fit);
        if (fit[lb] < best) { best = fit[lb]; best_sample = pop[lb]; }
        R.history[iter] = best;
    }
    R.best_gate_count = best;
    S.synthesize(best_sample, R.best_circuit);
    return R;
}

}  // namespace qbcs
