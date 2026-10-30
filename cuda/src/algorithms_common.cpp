// algorithms_common.cpp - helpers shared by the nine search algorithms + dispatcher.
#include "qbcs.h"
#include "evaluation.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace qbcs {

const char* algo_name(Algo a) {
    switch (a) {
        case Algo::AE_QTS: return "AE-QTS";
        case Algo::QTS:    return "QTS";
        case Algo::QEA:    return "QEA";
        case Algo::GA:     return "GA";
        case Algo::DE:     return "DE";
        case Algo::TS:     return "TS";
        case Algo::PSO:    return "PSO";
        case Algo::WOA:    return "WOA";
        case Algo::ABC:    return "ABC";
    }
    return "Other";
}

bool algo_from_id(int id, Algo& out) {
    if (id < 1 || id > 9) return false;
    out = static_cast<Algo>(id);
    return true;
}

void argsort_stable(const std::vector<int>& fitness, std::vector<int>& idx) {
    idx.resize(fitness.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return fitness[a] < fitness[b]; });
}

int evaluate(Synthesizer& S, const Sample& s, const Params& P, const std::vector<int>& target,
             long& failures) {
    int fitness = 0;
    evaluate_many(S, &s, 1, &fitness, P, target, failures);
    return fitness;
}

void evaluate_many(Synthesizer& S, const Sample* samples, std::size_t count, int* fitness,
                   const Params& P, const std::vector<int>& target, long& failures) {
    if (P.evaluator) P.evaluator->evaluate(samples, count, fitness);
    if (P.evaluator && !P.verify && !P.check_gpu) return;
    Circuit circuit;
    for (std::size_t i = 0; i < count; ++i) {
        const int reference = P.verify ? S.synthesize(samples[i], circuit) : S.gate_count(samples[i]);
        if (P.evaluator && fitness[i] != reference)
            throw std::runtime_error("accelerated fitness mismatch at sample " + std::to_string(i)
                + ": backend=" + std::to_string(fitness[i]) + ", CPU=" + std::to_string(reference));
        if (!P.evaluator) fitness[i] = reference;
        if (P.verify && !verify_circuit(circuit, S.encoding().n, target)) ++failures;
    }
}

void crossover_layer(const Encoding& E, int layer, const Sample& a, const Sample& b, int cp,
                     Sample& child) {
    // Python: new_layer = p1[:cp] + p2[cp:]  at the cycle (top-level list) granularity.
    const std::vector<int>& off = E.cyc_off(layer);
    const int split = off[cp];
    const int total = off[E.num_cycles];
    std::vector<uint8_t>& out = child.layer(layer);
    const std::vector<uint8_t>& la = a.layer(layer);
    const std::vector<uint8_t>& lb = b.layer(layer);
    out.resize(total);
    if (split > 0) std::memcpy(out.data(), la.data(), static_cast<size_t>(split));
    if (total - split > 0) std::memcpy(out.data() + split, lb.data() + split, static_cast<size_t>(total - split));
}

RunResult run_experiment(Algo algo, const Encoding& E, const Params& P, Rng& rng,
                         const std::vector<int>& target) {
    switch (algo) {
        case Algo::AE_QTS: return run_aeqts(E, P, rng, target);
        case Algo::QTS:    return run_qts(E, P, rng, target);
        case Algo::QEA:    return run_qea(E, P, rng, target);
        case Algo::GA:     return run_ga(E, P, rng, target);
        case Algo::DE:     return run_de(E, P, rng, target);
        case Algo::TS:     return run_ts(E, P, rng, target);
        case Algo::PSO:    return run_pso(E, P, rng, target);
        case Algo::WOA:    return run_woa(E, P, rng, target);
        case Algo::ABC:    return run_abc(E, P, rng, target);
    }
    return RunResult{};
}

}  // namespace qbcs
