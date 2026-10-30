// data_loader.cpp - see data_loader.h
#include "data_loader.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace qbcs {

// Strips '#' comment lines and, if present, an optional leading RevLib-style header line
// that holds only the bit count (e.g. the '14' on its own line in hwb14.txt, right after the
// two '# ...' copyright lines): a lone integer n immediately followed by exactly 2^n more
// values. That header line is otherwise indistinguishable from a legitimate first truth-table
// value, so it's only dropped when the rest of the file's value count actually matches 2^n;
// any other input (plain "v0 v1 ... " tables, single-line --tt input) passes through unchanged.
static std::string strip_comments_and_header(const std::string& text) {
    std::istringstream in(text);
    std::string line;
    std::vector<std::string> kept;
    while (std::getline(in, line)) {
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        kept.push_back(line.substr(first));
    }
    if (kept.empty()) return text;

    std::istringstream first_line(kept[0]);
    long long maybe_bits;
    std::string extra;
    if ((first_line >> maybe_bits) && !(first_line >> extra) && maybe_bits > 0 && maybe_bits < 63) {
        long long cnt = 0;
        for (size_t i = 1; i < kept.size(); ++i) {
            std::istringstream count_line(kept[i]);
            long long v;
            while (count_line >> v) ++cnt;
        }
        if (cnt == (1LL << maybe_bits)) kept.erase(kept.begin());
    }

    std::string joined;
    for (const auto& l : kept) { joined += l; joined += '\n'; }
    return joined;
}

static bool is_permutation(const std::vector<int>& t, int& bits, std::string& err) {
    const size_t m = t.size();
    bits = 0;
    while ((static_cast<size_t>(1) << bits) < m) ++bits;
    if ((static_cast<size_t>(1) << bits) != m) {
        err = "truth table length " + std::to_string(m) + " is not a power of two";
        return false;
    }
    std::vector<char> seen(m, 0);
    for (int v : t) {
        if (v < 0 || static_cast<size_t>(v) >= m || seen[v]) {
            err = "truth table is not a permutation of 0.." + std::to_string(m - 1) +
                  " (bad/duplicate value " + std::to_string(v) + ")";
            return false;
        }
        seen[v] = 1;
    }
    return true;
}

bool DataLoader::parse_table(const std::string& text, std::vector<int>& out, int& bits, std::string& err) {
    out.clear();
    std::string s = strip_comments_and_header(text);
    for (char& ch : s)
        if (ch == ',' || ch == '[' || ch == ']' || ch == ';') ch = ' ';
    std::istringstream in(s);
    long long v;
    while (in >> v) out.push_back(static_cast<int>(v));
    if (out.empty()) { err = "empty truth table"; return false; }
    return is_permutation(out, bits, err);
}

bool DataLoader::read_table_file(const std::string& path, std::vector<int>& out, int& bits, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::stringstream buf;
    buf << f.rdbuf();
    return parse_table(buf.str(), out, bits, err);
}

bool DataLoader::parse_sparse_table(const std::string& text, std::vector<int>& out, int& bits, std::string& err) {
    std::istringstream in(text);
    std::string line;
    std::vector<std::string> kept;
    while (std::getline(in, line)) {
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        kept.push_back(line.substr(first));
    }
    if (kept.empty()) { err = "empty sparse table"; return false; }

    {
        std::istringstream head(kept[0]);
        if (!(head >> bits) || bits <= 0 || bits >= 31) {
            err = "sparse table: first non-comment/non-blank line must be the bit count";
            return false;
        }
    }
    const size_t n = static_cast<size_t>(1) << bits;

    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<int>(i);  // identity default

    for (size_t li = 1; li < kept.size(); ++li) {
        std::string s = kept[li];
        for (char& ch : s) if (ch == ',') ch = ' ';
        std::istringstream ls(s);
        long long idx, val;
        while (ls >> idx >> val) {
            if (idx < 0 || static_cast<size_t>(idx) >= n || val < 0 || static_cast<size_t>(val) >= n) {
                err = "sparse table: index/value out of range (line " + std::to_string(li + 1) + ")";
                return false;
            }
            out[static_cast<size_t>(idx)] = static_cast<int>(val);
        }
        if (!ls.eof()) {
            err = "sparse table: trailing/unpaired token (line " + std::to_string(li + 1) + ")";
            return false;
        }
    }

    return is_permutation(out, bits, err);
}

bool DataLoader::read_sparse_table_file(const std::string& path, std::vector<int>& out, int& bits, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::stringstream buf;
    buf << f.rdbuf();
    return parse_sparse_table(buf.str(), out, bits, err);
}

bool DataLoader::load(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open problem library " + path; return false; }
    lib_.clear();
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            err = path + ":" + std::to_string(lineno) + ": expected '<bits> <idx> <name> : values'";
            return false;
        }
        std::istringstream head(line.substr(0, colon));
        Problem p;
        if (!(head >> p.bits >> p.idx)) {
            err = path + ":" + std::to_string(lineno) + ": bad header";
            return false;
        }
        head >> p.name;
        int bits = 0;
        std::string perr;
        if (!parse_table(line.substr(colon + 1), p.table, bits, perr) || bits != p.bits) {
            err = path + ":" + std::to_string(lineno) + ": " + (perr.empty() ? "bit width mismatch" : perr);
            return false;
        }
        lib_[p.bits].push_back(p);
    }
    for (auto& kv : lib_)
        std::sort(kv.second.begin(), kv.second.end(), [](const Problem& a, const Problem& b) { return a.idx < b.idx; });
    return true;
}

std::string DataLoader::info() const {
    std::string s;
    for (const auto& kv : lib_)
        s += std::to_string(kv.first) + "-bit: " + std::to_string(kv.second.size()) + " problems available\n";
    return s;
}

const Problem* DataLoader::get(int bits, int idx) const {
    auto it = lib_.find(bits);
    if (it == lib_.end()) return nullptr;
    for (const Problem& p : it->second)
        if (p.idx == idx) return &p;
    return nullptr;
}

}  // namespace qbcs
