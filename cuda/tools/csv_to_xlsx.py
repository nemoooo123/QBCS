"""Convert the .csv written by the C++ program into the .xlsx layout of the Python version.

    python csv_to_xlsx.py exp/13_bit/AE-QTS_Results/AE-QTS_13_1.csv          # one file
    python csv_to_xlsx.py exp                                               # whole tree

Requires pandas + openpyxl (same dependencies as the original project). The CSV already has
the exact DataFrame layout (index column, Gen_*, Execution_Time(s), Average_Convergence,
Std_Deviation), so this is a plain round-trip; plot_results.py can alternatively read the
CSV directly with pd.read_csv(path, index_col=0).
"""
import os
import sys

import pandas as pd


def convert(path: str) -> None:
    df = pd.read_csv(path, index_col=0)
    out = os.path.splitext(path)[0] + ".xlsx"
    df.to_excel(out)
    print(f"{path} -> {out}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for root, _, files in os.walk(arg):
                for f in files:
                    if f.endswith(".csv"):
                        convert(os.path.join(root, f))
        else:
            convert(arg)


if __name__ == "__main__":
    main()
