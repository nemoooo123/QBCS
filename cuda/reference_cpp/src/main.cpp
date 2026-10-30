// main.cpp - port of main.py
//
//   qbcs [num_bits] [problem_idx] [algo_id] [options]
//
// Runs `--experiments` independent trials of the chosen algorithm (in parallel threads),
// then writes exp/<n>_bit/<ALGO>_Results/<ALGO>_<n>_<idx>.txt  (summary, best circuit)
//                                         /<ALGO>_<n>_<idx>.csv  (Trial x Generation matrix)
// The CSV has the same layout as the pandas DataFrame the Python version wrote to .xlsx
// (rows Trial_i + Average_Convergence + Std_Deviation; columns Gen_j + Execution_Time(s)).
#include "qbcs.h"
#include "data_loader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

using namespace qbcs;

namespace {

void make_dirs(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || path[i] == '\\' || i + 1 == path.size()) MKDIR(cur.c_str());
    }
}

void usage() {
    std::cout <<
        "Usage: qbcs [num_bits] [problem_idx] [algo_id] [options]\n"
        "  algo_id: 1 AE-QTS  2 QTS  3 QEA  4 GA  5 DE  6 TS  7 PSO  8 WOA  9 ABC\n"
        "  (positional arguments omitted -> interactive prompts, like main.py)\n"
        "Options:\n"
        "  --experiments N   number of independent trials          (default 100)\n"
        "  --iterations N    generations per trial                 (default 10000)\n"
        "  --neighbors N     population / neighborhood size        (default 10)\n"
        "  --threads N       worker threads                        (default: all cores)\n"
        "  --seed S          base random seed (results reproducible for a given seed,\n"
        "                    independent of --threads)             (default: time-based)\n"
        "  --data FILE       problem library                       (default data/problems.txt)\n"
        "  --tt \"v0,v1,...\"  custom truth table given inline (ignores num_bits/problem_idx)\n"
        "  --tt-file FILE    custom truth table read from a text file (full 2^n-entry table)\n"
        "  --tt-sparse FILE  custom truth table read from a sparse file: first line is the\n"
        "                    bit count, remaining lines are \"index value\" pairs for the\n"
        "                    positions that differ from identity; everything else defaults\n"
        "                    to v(i)=i. Validated as a full permutation either way.\n"
        "  --out DIR         output root                           (default exp)\n"
        "  --verify          simulate every synthesized circuit and check it against the\n"
        "                    truth table (slow; for debugging)\n"
        "  --quiet           no per-experiment progress lines\n"
        "  --no-csv          skip writing the per-generation .csv (txt summary still written)\n"
        "Algorithm parameters (defaults = main.py):\n"
        "  --delta D (AE-QTS/QTS 0.01, QEA 0.0005)  --ga-k 5 --ga-pc 0.8 --ga-pm 0.005\n"
        "  --de-cr 0.01  --tabu 45  --pso-w 0.6 --pso-c1 3.0 --pso-c2 1.0  --woa-b 0.45\n"
        "  --abc-limit 40\n";
}

// Seed for experiment r: independent of thread scheduling.
uint64_t experiment_seed(uint64_t base, int r) {
    uint64_t x = base ^ (0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(r + 1));
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL; x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL; x ^= x >> 33;
    return x;
}

std::string fmt2(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2f", v);
    return buf;
}

// pandas/numpy-like float formatting for the CSV: integers print without decimals.
std::string fmt_num(double v) {
    if (std::isinf(v)) return "inf";
    if (std::isnan(v)) return "";
    if (v == std::floor(v) && std::fabs(v) < 1e15) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.1f", v);
        return buf;
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.10g", v);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    // ------------------------------------------------------------------ arguments -------
    std::vector<std::string> pos;
    int experiments = 100, threads = 0;
    uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    bool seed_given = false, quiet = false, no_csv = false;
    std::string data_path = "data/problems.txt", out_root = "exp", tt_inline, tt_file, tt_sparse;
    Params P;
    bool delta_given = false;
    double delta = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << what << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--experiments") experiments = std::stoi(need("--experiments"));
        else if (a == "--iterations")  P.max_iterations = std::stoi(need("--iterations"));
        else if (a == "--neighbors")   P.num_neighbors = std::stoi(need("--neighbors"));
        else if (a == "--threads")     threads = std::stoi(need("--threads"));
        else if (a == "--seed")      { seed = std::stoull(need("--seed")); seed_given = true; }
        else if (a == "--data")        data_path = need("--data");
        else if (a == "--tt")          tt_inline = need("--tt");
        else if (a == "--tt-file")     tt_file = need("--tt-file");
        else if (a == "--tt-sparse")   tt_sparse = need("--tt-sparse");
        else if (a == "--out")         out_root = need("--out");
        else if (a == "--verify")      P.verify = true;
        else if (a == "--quiet")       quiet = true;
        else if (a == "--no-csv")      no_csv = true;
        else if (a == "--delta")     { delta = std::stod(need("--delta")); delta_given = true; }
        else if (a == "--ga-k")        P.ga_k = std::stoi(need("--ga-k"));
        else if (a == "--ga-pc")       P.ga_pc = std::stod(need("--ga-pc"));
        else if (a == "--ga-pm")       P.ga_pm = std::stod(need("--ga-pm"));
        else if (a == "--de-cr")       P.de_cr = std::stod(need("--de-cr"));
        else if (a == "--tabu")        P.ts_tabu_size = std::stoi(need("--tabu"));
        else if (a == "--pso-w")       P.pso_w = std::stod(need("--pso-w"));
        else if (a == "--pso-c1")      P.pso_c1 = std::stod(need("--pso-c1"));
        else if (a == "--pso-c2")      P.pso_c2 = std::stod(need("--pso-c2"));
        else if (a == "--woa-b")       P.woa_b = std::stod(need("--woa-b"));
        else if (a == "--abc-limit")   P.abc_limit = std::stoi(need("--abc-limit"));
        else if (!a.empty() && a[0] == '-') { std::cerr << "unknown option " << a << "\n"; usage(); return 2; }
        else pos.push_back(a);
    }
    if (delta_given) P.delta_theta_aeqts = P.delta_theta_qts = P.delta_theta_qea = delta;

    // ------------------------------------------------------------------ problem ----------
    DataLoader loader;
    std::string err;
    const bool custom = !tt_inline.empty() || !tt_file.empty() || !tt_sparse.empty();
    if (!custom) {
        if (!loader.load(data_path, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
        std::cout << loader.info();
    }

    int num_bits = 0, problem_idx = 0, algo_id = 0;
    auto prompt_int = [](const char* msg, int& v) {
        std::cout << msg;
        std::string line;
        if (!std::getline(std::cin, line)) return false;
        try { v = std::stoi(line); } catch (...) { return false; }
        return true;
    };
    if (custom) {
        if (pos.size() >= 1) algo_id = std::stoi(pos[0]);
        else if (!prompt_int("Select Algorithm (1: AE-QTS, 2: QTS, 3: QEA, 4: GA, 5: DE, 6: TS, 7: PSO, 8: WOA, 9: ABC): ", algo_id)) {
            std::cout << "Invalid input. Please enter numeric values.\n"; return 1;
        }
    } else {
        if (pos.size() >= 3) { num_bits = std::stoi(pos[0]); problem_idx = std::stoi(pos[1]); algo_id = std::stoi(pos[2]); }
        else if (!prompt_int("Enter number of bits (n): ", num_bits) ||
                 !prompt_int("Enter problem index: ", problem_idx) ||
                 !prompt_int("Select Algorithm (1: AE-QTS, 2: QTS, 3: QEA, 4: GA, 5: DE, 6: TS, 7: PSO, 8: WOA, 9: ABC): ", algo_id)) {
            std::cout << "Invalid input. Please enter numeric values.\n"; return 1;
        }
    }
    Algo algo;
    if (!algo_from_id(algo_id, algo)) { std::cerr << "Error: algorithm id must be 1..9\n"; return 1; }

    std::vector<int> target;
    std::string problem_name;
    if (custom) {
        bool ok = !tt_sparse.empty() ? DataLoader::read_sparse_table_file(tt_sparse, target, num_bits, err)
                : !tt_file.empty()   ? DataLoader::read_table_file(tt_file, target, num_bits, err)
                                     : DataLoader::parse_table(tt_inline, target, num_bits, err);
        if (!ok) { std::cerr << "Error: " << err << "\n"; return 1; }
        problem_idx = 0;
        problem_name = "custom";
    } else {
        const Problem* p = loader.get(num_bits, problem_idx);
        if (!p) { std::cout << "Error: Targeted problem not found. Please restart.\n"; return 1; }
        target = p->table;
        problem_name = p->name;
    }

    std::cout << "\nTarget Truth Table (" << num_bits << "-bit, " << problem_name << "): [";
    for (size_t i = 0; i < target.size(); ++i) std::cout << (i ? ", " : "") << target[i];
    std::cout << "]\n";

    // ------------------------------------------------------------------ setup ------------
    std::vector<std::vector<int>> cycles = find_cycles(target);
    const bool identity = cycles.empty();
    Encoding E;
    if (!identity) E = build_encoding(cycles, num_bits);

    if (threads <= 0) threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) threads = 1;
    threads = std::min(threads, std::max(1, experiments));

    std::cout << "Algorithm: " << algo_name(algo) << " | cycles: " << E.num_cycles
              << " | Layer sizes L1/L2/L3/L4 = " << E.l1_size << "/" << E.l2_size << "/" << E.l3_size << "/" << E.l4_size
              << "\nStarting " << experiments << " experiments, each with " << P.max_iterations
              << " iterations (" << threads << " threads, seed " << seed << (seed_given ? "" : " [time-based]") << ")...\n";

    // ------------------------------------------------------------------ run ---------------
    std::vector<RunResult> results(experiments);
    std::vector<double> exec_times(experiments, 0.0);
    std::atomic<int> next_exp{0};
    std::mutex io_mutex;
    const auto t_all0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        for (;;) {
            const int r = next_exp.fetch_add(1);
            if (r >= experiments) break;
            const auto t0 = std::chrono::steady_clock::now();
            RunResult res;
            if (identity) {
                res.best_gate_count = 0;
                res.history.assign(P.max_iterations, 0.0);
            } else {
                Rng rng(experiment_seed(seed, r));
                try {
                    res = run_experiment(algo, E, P, rng, target);
                } catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lk(io_mutex);
                    std::cerr << "Experiment " << r + 1 << " failed: " << ex.what() << "\n";
                    std::exit(1);
                }
            }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            exec_times[r] = dt;
            results[r] = std::move(res);
            if (!quiet) {
                std::lock_guard<std::mutex> lk(io_mutex);
                if (identity)
                    std::cout << "Experiment " << r + 1 << ": Identity mapping detected (Zero gates required).\n";
                else
                    std::cout << "--- Experiment " << r + 1 << " Finished | Best Gate: " << results[r].best_gate_count
                              << " | Time: " << fmt2(dt) << "s"
                              << (P.verify ? (" | verify failures: " + std::to_string(results[r].verify_failures)) : "")
                              << " ---\n";
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all0).count();

    // ------------------------------------------------------------------ statistics -------
    const int G = P.max_iterations;
    std::vector<double> mean_curve(G, 0.0), std_pop(G, 0.0), std_sample(G, 0.0);
    for (int g = 0; g < G; ++g) {
        double s = 0.0;
        for (int r = 0; r < experiments; ++r) s += results[r].history[g];
        const double m = s / experiments;
        double ss = 0.0;
        for (int r = 0; r < experiments; ++r) { const double d = results[r].history[g] - m; ss += d * d; }
        mean_curve[g] = m;
        std_pop[g] = std::sqrt(ss / experiments);                                   // np.std (ddof=0)
        std_sample[g] = experiments > 1 ? std::sqrt(ss / (experiments - 1)) : 0.0;  // pandas .std() (ddof=1)
    }
    int best_exp = 0;
    for (int r = 1; r < experiments; ++r)
        if (results[r].best_gate_count < results[best_exp].best_gate_count) best_exp = r;
    const int global_min = results[best_exp].best_gate_count;
    double time_sum = 0.0;
    for (double t : exec_times) time_sum += t;
    const double avg_time = experiments ? time_sum / experiments : 0.0;
    long total_verify_failures = 0;
    for (const auto& r : results) total_verify_failures += r.verify_failures;

    // ------------------------------------------------------------------ files ------------
    const std::string save_dir = out_root + "/" + std::to_string(num_bits) + "_bit/" + algo_name(algo) + "_Results";
    make_dirs(save_dir);
    const std::string base = std::string(algo_name(algo)) + "_" + std::to_string(num_bits) + "_" + std::to_string(problem_idx);
    const std::string txt_path = save_dir + "/" + base + ".txt";
    const std::string csv_path = save_dir + "/" + base + ".csv";

    {
        std::ofstream f(txt_path, std::ios::binary);  // UTF-8 (the Python version wrote UTF-16)
        f << "========================================\n";
        f << "      FINAL EXPERIMENTAL STATISTICS      \n";
        f << "========================================\n";
        f << "Best Gate Counts per Trial: [";
        for (int r = 0; r < experiments; ++r) f << (r ? ", " : "") << results[r].best_gate_count;
        f << "]\n";
        f << "Global Minimum Gate Count:  " << global_min << "\n";
        f << "Final Result (Gen " << G << "): " << fmt2(mean_curve.empty() ? 0.0 : mean_curve.back())
          << " \xC2\xB1 " << fmt2(std_pop.empty() ? 0.0 : std_pop.back()) << "\n";
        f << "Average Time per Experiment: " << fmt2(avg_time) << "s\n";
        f << "Total Wall Time: " << fmt2(wall) << "s (" << threads << " threads)\n";
        if (P.verify) f << "Circuit Logic Verification Failures: " << total_verify_failures << "\n";
        f << std::string(40, '-') << "\n";
        f << "Best Circuit Structure:\n" << circuit_to_python_str(results[best_exp].best_circuit, num_bits) << "\n";
        f << "========================================\n";
    }
    if (!no_csv) {
        std::ofstream f(csv_path, std::ios::binary);
        f << "";
        for (int g = 0; g < G; ++g) f << ",Gen_" << g + 1;
        f << ",Execution_Time(s)\n";
        for (int r = 0; r < experiments; ++r) {
            f << "Trial_" << r + 1;
            for (int g = 0; g < G; ++g) f << ',' << fmt_num(results[r].history[g]);
            f << ',' << fmt_num(exec_times[r]) << "\n";
        }
        f << "Average_Convergence";
        for (int g = 0; g < G; ++g) f << ',' << fmt_num(mean_curve[g]);
        f << ',' << fmt_num(avg_time) << "\n";
        f << "Std_Deviation";
        for (int g = 0; g < G; ++g) f << ',' << fmt_num(std_sample[g]);
        f << ",\n";
    }

    std::cout << "\nGlobal Minimum Gate Count: " << global_min
              << " | Final Result (Gen " << G << "): " << fmt2(mean_curve.empty() ? 0.0 : mean_curve.back())
              << " \xC2\xB1 " << fmt2(std_pop.empty() ? 0.0 : std_pop.back())
              << " | Avg time/experiment: " << fmt2(avg_time) << "s | Wall: " << fmt2(wall) << "s\n";
    if (P.verify) std::cout << "[Verify] circuit logic failures: " << total_verify_failures << "\n";
    std::cout << "[System] Summary report saved to: " << txt_path << "\n";
    if (!no_csv)
        std::cout << "[System] Full convergence history with execution times saved to: " << csv_path << "\n";
    return 0;
}
