# Original C++ Reference

`src/`, `CMakeLists.txt`, and `tests/replay_test.cpp` in this directory come from the uploaded `qbcs_cpp.zip` and are used for independent regression. The only change is the removal of the ROT family (IDs 11–13), matching `../../cpp/src`. The SHA-256 hashes in `../validation/source_manifest.json` were recorded before that change.

Build: from the `cuda` directory, run `cmake -S reference_cpp -B build-reference`, then `cmake --build build-reference --config Release --parallel`.

The problem table is shared with the main project's `data/problems.txt`; when running the reference executable directly, pass `--data data/problems.txt`. `regression.py` passes the full data path automatically.
