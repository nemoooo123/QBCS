# 等價驗證與效能測量

測試分為三種證據：原始 Python 樣本重播、原始 C++ 執行結果逐項比對，以及共享整數核心的測試。`emulated` 在 CPU 執行共享整數核心，不能取代 NVIDIA GPU 上的 CUDA 測試。

## 1. 原始 C++ 與新執行檔：完整迭代歷史比對

先把上傳的原始 C++ 專案另外編譯為獨立執行檔，保持其來源未修改。在本專案根目錄執行：

```bash
python3 tests/regression.py \
  --baseline /absolute/path/to/original/qbcs \
  --candidate build/qbcs_cuda \
  --backends cpu,emulated \
  --work-dir regression_cpu_emulated
```

Windows 可使用 `python`，執行檔路徑則改為實際的 `.exe`。所有 Python 測試與 benchmark 控制工具只依賴標準函式庫；產生 Python 重播紀錄時才需要原始 Python 專案的依賴。

預設測試矩陣：

| 項目 | 涵蓋內容 |
|---|---|
| 演算法 | ID 1–10 全部 |
| 問題 | 3-bit、5-bit、identity、最大 Hamming distance 的單一 swap、全為 distance-1 的 marker steps |
| 族群 | N = 4、5、10，涵蓋奇數與偶數 |
| 更新量 | 原始預設值、`--delta 0`、`--delta 1` |
| 停滯/記憶 | `--abc-limit 1`、`--tabu 2` |
| 執行緒 | 1 與 3，固定每組 trial seed |
| GPU 選項 | batch = 17；CUDA 跑 thread 與 warp decoder；非 CPU 模式加 `--check-gpu` |
| 正確性 | 每次均加 `--verify`，檢查所有實際評估電路的真值表 |

共 9 組案例 × 10 種演算法。CPU 與 emulated 各有 2 個執行緒設定；emulated 只執行共享 scalar 核心，因此即使 `--gpu-decodes` 包含 warp，也只跑一次 thread 模式。預設 `cpu,emulated` 合計 **360 次候選結果比對**，另有 90 次原版參照執行。CUDA 模式才有 2 × 2 個執行緒/decoder 設定，單獨 `--backends cuda` 亦為 360 次候選比對。

每次比對完整 CSV，包括所有 trial 的每一代值、平均與標準差；只排除 `Execution_Time(s)`。此外逐項比對每個 trial 的最終 best count、全域最小值、最終統計值，以及報告中的**整份全域最佳 gate list，包含 gate 順序**。原版輸出未包含每個 trial 各自的最佳 gate list，因此此 CLI 測試不宣稱逐一比對未輸出的資料。

### 在 NVIDIA GPU 上執行

```bash
python3 tests/regression.py \
  --baseline /absolute/path/to/original/qbcs \
  --candidate build/qbcs_cuda \
  --backends cuda \
  --gpu-decodes thread,warp \
  --work-dir regression_cuda
```

CUDA 執行失败會報錯，測試不會自行改用 CPU。`--suite smoke` 只跑前三個案例，但保留全部 10 種演算法。要另外檢查一個族群被拆成多個 batch：

```bash
python3 tests/regression.py \
  --baseline /absolute/path/to/original/qbcs \
  --candidate build/qbcs_cuda \
  --backends cuda --gpu-batch 1 --suite smoke \
  --work-dir regression_cuda_batch1
```

`--work-dir` 保留每次的完整 argv、stdout、stderr、CSV、TXT 與 `regression_report.json`。`--report PATH` 可另外輸出精簡 JSON 報告。參照或候選任一結果不一致時，程式回傳非零 exit code。

## 2. 分批 trial / 多張 GPU 的 seed 一致性

```bash
python3 tests/shard_regression.py \
  --baseline /absolute/path/to/original/qbcs \
  --candidate build/qbcs_cuda --backend emulated --gpu-decode thread \
  --work-dir regression_shards
```

先比對原版與新版的 6-trial 執行，再用 `--trial-offset` 拆成 2 + 1 + 3 個 trial。合併後的每代歷史、各 trial best count，以及採相同 tie-breaking 的全域最佳 gate list 都必須相同。N = 5、GPU batch = 3，同時涵蓋單次族群切批。GPU 測試請改成 `--backend cuda`。

## 3. 原始 Python → C++ 重播

本專案保留上傳 C++ 專案原有的 `dump_replay.py` 和 `replay_test.cpp`。可用包裝工具產生並重播：

```bash
python3 tests/python_replay.py \
  --replay-binary build/replay_test \
  --python-root /absolute/path/to/original/python/project \
  --output python_replay_results
```

或重播既有 trace：

```bash
python3 tests/python_replay.py --replay-binary build/replay_test \
  --traces /path/to/trace_3_aeqts.txt /path/to/trace_5_qts.txt
```

此測試將原始 Python 產生的相同四層 bit 樣本送進 C++，比對每個電路的 gate list、fitness 排名，以及 AE-QTS/QTS/QEA 的更新後機率。重播測試內的 update helper 來自原始 C++ 測試副本；它本身不能證明新版 production driver 的更新正確，所以必須搭配第 1 項的實際執行檔比對。Python 與 C++ 本來使用不同亂數產生器，不能把相同 seed 當成跨語言相同樣本。

## 4. CPU / CUDA 效能 sweep

```bash
python3 tools/benchmark.py --binary build/qbcs_cuda \
  --bits 13 --problem 1 --algorithm 1 \
  --experiments 32 --iterations 200 --neighbors 10 \
  --threads 1,4,16 --gpu-batches 32,128,512 \
  --gpu-decodes thread,warp --out benchmark_13bit
```

也可加 `--baseline /path/to/original/qbcs`，直接對原始 C++ 計時。未指定時以新版的 `--backend cpu` 為基準。自訂問題可用 `--tt`、`--tt-file` 或 `--tt-sparse`；`--delta` 的單位仍依原演算法定義。

每組設定預設先跑 1 次不計時 warmup，再測 3 次，報告中位數、最快值、最慢值。計時涵蓋整個執行程序，包括 CUDA 啟動、資料準備、運算及輸出；warmup 可以暖機，但每次正式執行仍重新啟動程序。benchmark 不開啟 `--verify` / `--check-gpu`，避免把逐電路驗證成本算成演算法正常執行時間；每次完成後仍會和 CPU 基準**精確比對全部收斂歷史與全域最佳電路**，有任何差異立即中止。

輸出 `benchmark.json`、`benchmark_summary.csv` 及所有原始紀錄；同時提供相同 thread 數 CPU 的加速比，以及**所有已測 CPU 設定中最快者**的加速比。不得只挑 CPU 單執行緒作為多執行緒 CUDA 配置的唯一比較基準。

先完成 CUDA correctness regression 再進行正式 benchmark。建議保留 GPU 名稱、driver、CUDA compiler、CPU 型號、測试命令，並在目標電腦上關閉會明顯競爭 CPU/GPU 的工作。沒有實測前不填入加速倍數。`--backend emulated` 僅用於測試 benchmark 腳本，輸出會明確標示不是 GPU 量測。
