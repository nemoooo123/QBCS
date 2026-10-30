import os


class DataLoader:
    """
    Runtime loader for the reversible-circuit problem library.

    Mirrors the C++ port's design (qbcs_cpp/src/data_loader.cpp): problems live in a plain
    text catalog file instead of being embedded as Python literals, so the library can grow
    (new bit widths, new problems) without bloating this module or requiring a code change.

    Catalog format (data/problems.txt, one line per problem):
        # comment lines start with '#'
        <num_bits> <problem_idx> <name> : v0 v1 v2 ... v(2^n - 1)
    """

    DEFAULT_CATALOG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "problems.txt"
    )

    def __init__(self, catalog_path=None):
        self.catalog_path = catalog_path or self.DEFAULT_CATALOG_PATH
        self.data_map = {}   # bits -> {idx: table}
        self.names = {}      # bits -> {idx: name}
        self._load_catalog()

    def _load_catalog(self):
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                head, sep, value_part = line.partition(":")
                if not sep:
                    raise ValueError(f"{self.catalog_path}:{lineno}: expected '<bits> <idx> <name> : values'")
                head_tokens = head.split()
                if len(head_tokens) < 2:
                    raise ValueError(f"{self.catalog_path}:{lineno}: bad header")
                bits, idx = int(head_tokens[0]), int(head_tokens[1])
                name = head_tokens[2] if len(head_tokens) > 2 else ""
                table = [int(x) for x in value_part.split()]
                if len(table) != 2 ** bits:
                    raise ValueError(
                        f"{self.catalog_path}:{lineno}: {name}: table length {len(table)} != 2^{bits}"
                    )
                self.data_map.setdefault(bits, {})[idx] = table
                self.names.setdefault(bits, {})[idx] = name

    def get_info(self):
        """Display information about available data in the library."""
        for bit, data in sorted(self.data_map.items()):
            print(f"{bit}-bit: {len(data)} problems available")

    def get_output(self, n_bit, choice):
        """Retrieve target output based on bit size and index selection."""
        if n_bit in self.data_map:
            if choice in self.data_map[n_bit]:
                return self.data_map[n_bit][choice]
            print(f"Error: {n_bit}-bit library does not have problem #{choice}")
        else:
            print(f"Error: {n_bit}-bit data is currently not supported")
        return None

    @staticmethod
    def parse_sparse_table(text):
        """
        Parse a sparse permutation spec: only the positions that differ from the identity
        (v(i) != i) need to be listed, as "index value" pairs - convenient once 2^n gets too
        large to hand-write a full table for. Format ('#' comments and blank lines ignored):
            <num_bits>
            i0 v0
            i1 v1
            ...
        Every unlisted index defaults to v(i)=i. Returns the expanded full table (list[int]);
        raises ValueError if the first non-comment line isn't the bit count, a pair is
        malformed/out of range, or the expanded result isn't a valid permutation.
        """
        kept = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            kept.append(line)
        if not kept:
            raise ValueError("empty sparse table")

        bits = int(kept[0])
        n = 1 << bits
        table = list(range(n))  # identity default

        for lineno, raw in enumerate(kept[1:], start=2):
            tokens = raw.replace(",", " ").split()
            if len(tokens) % 2 != 0:
                raise ValueError(f"sparse table: trailing/unpaired token (line {lineno})")
            values = [int(x) for x in tokens]
            for i in range(0, len(values), 2):
                idx, val = values[i], values[i + 1]
                if not (0 <= idx < n) or not (0 <= val < n):
                    raise ValueError(f"sparse table: index/value out of range (line {lineno})")
                table[idx] = val

        if sorted(table) != list(range(n)):
            raise ValueError("sparse table: result is not a valid permutation (value collision)")
        return table

    @staticmethod
    def read_sparse_table_file(path):
        """Read and parse a sparse truth table file (see parse_sparse_table)."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return DataLoader.parse_sparse_table(text)