# 原始 C++ 參照

本目錄的 src/、CMakeLists.txt、LICENCE 與 tests/replay_test.cpp 原封保留自上傳的 qbcs_cpp.zip，供獨立 regression 使用。檔案 SHA-256 見 ../validation/source_manifest.json。

編譯：從 CUDA 專案根目錄執行 `cmake -S reference_cpp -B build-reference`，接著 `cmake --build build-reference --config Release --parallel`。

資料表共用主專案的 data/problems.txt，直接跑參照執行檔時傳入 `--data data/problems.txt`。regression.py 會自動指定完整資料路徑。
