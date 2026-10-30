# Original C++ Reference

`src/`, `CMakeLists.txt`, and `tests/replay_test.cpp` in this directory come from the uploaded `qbcs_cpp.zip` and are used for independent regression. They are kept identical to `../../cpp/src`. Changes from the upload: the extra variants AE-QTS-GB and ROT (IDs 10–13) were removed, and `repair_l4_step` no longer uses a `thread_local` buffer, which crashed multithreaded MinGW builds at thread exit (results are unchanged). The SHA-256 hashes in `../validation/source_manifest.json` were recorded before these changes.

Build: from the `cuda` directory, run `cmake -S reference_cpp -B build-reference`, then `cmake --build build-reference --config Release --parallel`.

The problem table is shared with the main project's `data/problems.txt`; when running the reference executable directly, pass `--data data/problems.txt`. `regression.py` passes the full data path automatically.
