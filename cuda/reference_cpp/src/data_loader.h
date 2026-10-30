// data_loader.h - benchmark library (port of utils/data_loader.py)
//
// The truth tables live in a plain text file (default: data/problems.txt) instead of a
// Python dictionary, so custom problems can be added without recompiling:
//
//     # comment lines start with '#'
//     <num_bits> <problem_idx> <name> : v0 v1 v2 ... v(2^n - 1)
//
// Values are the permutation truth table (output for input i at position i).
#pragma once

#include <map>
#include <string>
#include <vector>

namespace qbcs {

struct Problem {
    int bits = 0;
    int idx = 0;
    std::string name;
    std::vector<int> table;
};

class DataLoader {
public:
    // Load the library; returns false (with message) if the file cannot be read.
    bool load(const std::string& path, std::string& err);
    // get_info(): "n-bit: k problems available"
    std::string info() const;
    // get_output(): 1-based problem index; nullptr if missing
    const Problem* get(int bits, int idx) const;
    // Parse a truth table given inline ("21,6,3,0,...") or from a text file (whitespace/comma
    // separated). Returns false with message if it is not a permutation of 0..2^n-1.
    static bool parse_table(const std::string& text, std::vector<int>& out, int& bits, std::string& err);
    static bool read_table_file(const std::string& path, std::vector<int>& out, int& bits, std::string& err);

    // Sparse format: only the positions that differ from the identity (v(i) != i) need to be
    // listed, as "index value" pairs - convenient for a permutation that's mostly a no-op and
    // only moves a handful of states, especially once 2^n gets too large to hand-write a full
    // table for. Format ('#' comments allowed, blank lines ignored):
    //     <num_bits>
    //     i0 v0
    //     i1 v1
    //     ...
    // Every unlisted index defaults to v(i)=i; the result is validated as a full permutation
    // (an index and a value each still resolve to exactly one slot) exactly like parse_table.
    static bool parse_sparse_table(const std::string& text, std::vector<int>& out, int& bits, std::string& err);
    static bool read_sparse_table_file(const std::string& path, std::vector<int>& out, int& bits, std::string& err);

private:
    std::map<int, std::vector<Problem>> lib_;
};

}  // namespace qbcs
