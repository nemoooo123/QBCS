# QBCS CUDA：保留原搜尋邏輯的 GPU 版本

本專案由提供的 Python／C++ 量子布林電路合成程式改寫，採 **C++ 控制搜尋、CUDA 批次合成與計算 fitness** 的架構。保留原 C++ 的 10 種演算法（ID 1–10；原 C++ 的 ID 11–13 ROT 系列已移除）、四層編碼、RNG、Layer 4 修復、排序與更新順序。新增的平行化作用於彼此獨立的候選解與獨立 trial。

**交付狀態：真正的 NVCC 13.0.88／`sm_120` 編譯與連結已通過，CPU 上的等價驗證已通過；目前環境沒有可用 NVIDIA GPU，CUDA kernel 的實機正確性、Compute Sanitizer 與加速倍數尚未量測。** 詳細證據在 `validation/`、`tests/validation/` 與 `docs/VALIDATION.md`。執行時若 CUDA 不可用會回報錯誤；可用 `--backend cpu` 明確選擇 CPU。

## 1. 快速開始

需要 C++17 編譯器、CMake 3.24 以上，以及 NVIDIA CUDA Toolkit。RTX 5090／5080 的 compute capability 為 12.0，編譯參數使用 `120`；建議安裝 CUDA Toolkit 12.8 以上及相容顯示卡驅動。本次實際編譯使用 CUDA 13.0.88。GPU 架構依據 [NVIDIA GPU capability 表](https://developer.nvidia.com/cuda/gpus)，Toolkit 的架構支援請參照 [NVIDIA CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)。

### Windows：Visual Studio 2022 的 x64 開發者命令列／PowerShell

安裝 Visual Studio 的 C++ desktop development 與 CUDA 的 Visual Studio integration，進入解壓縮後的 `qbcs_cuda` 目錄：

```powershell
cmake -S . -B build -A x64 -DQBCS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

先跑小問題，逐一比對 GPU fitness 與原 CPU 合成器，並驗證真值表：

```powershell
.\build\Release\qbcs_cuda.exe 5 1 1 --backend cuda --experiments 3 --iterations 20 --neighbors 10 --threads 3 --seed 2026 --check-gpu --verify --out exp_check
```

正式執行的起始設定：

```powershell
.\build\Release\qbcs_cuda.exe 13 1 1 --backend cuda --experiments 32 --iterations 200 --neighbors 10 --threads 8 --gpu-batch 128 --seed 2026 --out exp_cuda
```

這是測量用的起始設定。請依原實驗規格設定 `--experiments`、`--iterations`、`--neighbors` 與 `--delta`；程式未加入額外停止準則或改變搜尋空間。

### Linux／WSL

```bash
cmake -S . -B build -DQBCS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/qbcs_cuda 5 1 1 --backend cuda --experiments 3 --iterations 20 --neighbors 10 --threads 3 --seed 2026 --check-gpu --verify --out exp_check
```

`ctest` 執行共用核心及排程服務的 CPU 測試；上面的 CUDA 執行命令才會實際啟動 GPU。Windows 的執行檔路徑與 Linux 不同，其餘 CLI 參數相同。Windows build 尚未在本次環境實測。

若使用其他顯示卡，依其 compute capability 調整 `CMAKE_CUDA_ARCHITECTURES`，例如 RTX 4090 使用 `89`。未指定時本專案預設編譯 `86` 的 SASS 與 PTX；PTX 相容性見 [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)。

### 不安裝 CUDA 的參照／驗證版本

```bash
cmake -S . -B build-cpu -DQBCS_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
./build-cpu/qbcs_cuda 5 1 1 --backend cpu --experiments 3 --iterations 20 --seed 2026
./build-cpu/qbcs_cuda 5 1 1 --backend emulated --experiments 3 --iterations 20 --seed 2026 --check-gpu
```

`emulated` 在 CPU 執行與 thread CUDA kernel 共用的整數核心，用於定位解码、打包、記憶體配置與排程問題。它不模擬 GPU 效能，也不執行 warp intrinsics。CUDA 與 CPU build 共用相同搜尋 driver。

## 2. 哪些邏輯保持一致

| 項目 | 本版本處理 |
|---|---|
| Cycle decomposition | 保留原始分解、反向與閉合方式 |
| Layer 1 | MSB-first 解碼；priority 降序；同值保留原 cycle index 順序 |
| Layer 2 | 保留 `% mapping`；2-cycle 仍只用原本那一步；不修改斷邊選擇 |
| Layer 3 | 完整編碼搜尋；priority 同值時仍按原 MSB-first index 排序 |
| Layer 4 | 保留原抽樣、原 RNG 次序與原 Fisher–Yates 修復；相同 head/tail/pivot 閘序 |
| Gate reduction | 原 `A*A=I` 與 moving rule，依整條 emission order 執行，含跨 cycle 的抵消 |
| Fitness | 約簡後 gate count，保持整數與最小化目標 |
| Ranking | 保留 stable fitness sort；同 fitness 的配對順序不變 |
| AE-QTS 更新 | 每個 pair 按原 rank 依序以 `delta/(r+1)` 更新，並在每一次更新後處理邊界 |
| Best tracking | 保留原 strict improvement、歷史最佳與各演算法的接受條件 |
| Randomness | 原 C++ xoshiro256++；相同 trial seed 與亂數呼叫順序 |
| 實驗輸出 | 原 TXT／CSV 及每代 best-so-far 歷史，增加 backend／batch／分階段計時資訊 |

Python 與 C++ 原本就使用不同 RNG 和不同抽樣呼叫順序。因此本專案採用兩種對照：Python 的固定樣本重播，以及原 C++ 的相同 seed 逐代比對。後者是 CUDA 版本的執行軌跡標準；跨編譯器／跨數學函式庫仍應重新執行 regression，尤其是 PSO 等浮點運算較多的演算法。編譯時未使用 fast math，GPU fitness 核心全部使用整數。

### 演算法 ID

| ID | 名稱 | 保留的核心規則 |
|---|---|---|
| 1 | AE-QTS | 當代 harmonic best–worst 配對，直接機率增減 |
| 2 | QTS | 當代單一 best–worst，直接機率增減 |
| 3 | QEA | 朝歷史最佳更新每個 bit |
| 4 | GA | 原 tournament／cycle crossover／mutation／elitism |
| 5 | DE | 原 binary mutation／crossover／`<=` 接受與同步換代 |
| 6 | TS | 原複合 move、tabu 與 aspiration |
| 7 | PSO | 原連續位置／速度、離散化與 pbest／gbest |
| 8 | WOA | 原代內立即更新 pop／leader |
| 9 | ABC | 原 scout → employed → onlooker 與原地更新 |
| 10 | AE-QTS-GB | 原歷史最佳 harmonic 機率更新 |

**原程式的 ID 1 並不是角度旋轉版本。** 對 ID 1／2／3／10，`--delta` 是直接機率更新量。完整稽核見 `docs/REFERENCE_AUDIT.md`。

## 3. 加速架構

CPU 按原順序產生／修復候選解。多個獨立 trial 把可獨立評估的樣本交給共同排程服務；服務將請求合併為 GPU batch。CPU 解出 L1／L2 的原始 schedule，GPU 按 `sample × active step` 平行執行 L3／L4，建立整條未約簡閘序，再以每個 sample 一條順序鏈完成 moving-rule reduction。每次只把 fitness 整數送回 CPU；最後的最佳電路由原 CPU 合成器重建一次。

具体結構改善：

- Topology metadata 建立一次，常駐 GPU；候選解與 scratch buffers 重用。
- Layer 3 的排序 key 同時編碼 priority 與原 tie index，thread decoder 只需一個小型 `path[]`，沿用同一陣列儲存已解碼的 edge。
- 單一 raw gate 壓成 32-bit 整數；reducer 的每個 predecessor link 也是 32-bit。保留先後關係所需資料，降低 scratch 空間。
- Schedule、raw gates 與 predecessor buffers 使用 `[item][batch]` 排列，讓相鄰 GPU lanes 處理不同樣本時盡量取得連續記憶體。
- Host staging 使用 pinned memory 與非同步 CUDA copy；GPU 工作可與其他 trial 的 CPU 抽樣／更新同時進行。每一個 batch 的 GPU stages 仍在同一 stream 依序執行。
- 同一族群若超過 batch capacity，服務自動分批，全部完成後才讓該 trial 更新。batch 大小不等於族群 N。

這些策略依循 NVIDIA 關於減少傳輸、常駐資料及記憶體存取的 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。架構推導、記憶體公式與效能限制見 `docs/CUDA_DESIGN.md`。

WOA 與 ABC 的代內依賴保留為逐個評估，僅把不同 trial 的獨立請求合併。GA／DE／TS／PSO 與 quantum 系列則能批次處理一整組獨立候選解。這個差異可能使各演算法得到不同加速幅度。

## 4. GPU 參數與記憶體

| 選項 | 預設 | 用途 |
|---|---:|---|
| `--backend` | `cuda` | `cuda`、原 `cpu`、或共用核心 `emulated` |
| `--threads` | CPU 邏輯核心數，上限為 trial 數 | 同時進行的獨立 CPU trial；與 N 分開 |
| `--gpu-batch` | 128 | 單 batch 最多幾個候選解；不足時照實際數量執行 |
| `--gpu-memory-mb` | 0 | GPU 陣列預算，單位 MiB；0 使用當時 free VRAM 的約 80% |
| `--gpu-device` | 0 | 本程序使用哪一張 GPU |
| `--gpu-wait-us` | 200 | 等待其他 trial 合併 batch 的時間上限；可設 0 |
| `--gpu-decode` | `thread` | `thread` 或 `warp`，由實測決定較快者 |
| `--check-gpu` | 關 | 每個加速 fitness 都和原 CPU fitness 比對 |
| `--verify` | 關 | 重建每個已評估電路並驗證完整真值表；成本較高 |
| `--trial-offset` | 0 | 全域 trial 起點，供多程序／多機器切分相同實驗 |

程式按實際 topology 計算每個樣本所需 VRAM，再自動降低 batch capacity 以符合預算。**batch 調小不會調小族群 N。** GPU 預算涵蓋主要 device arrays，另需 CPU 族群／機率陣列、pinned staging、CUDA context 與其他 runtime 開銷。較大問題請也留意系統 RAM；`--threads` 過大會同時保存許多份 trial 狀態。

由附帶資料表直接計算的配置範例：

| 問題 | 每樣本 device arrays | capacity 128，加共享 metadata |
|---|---:|---:|
| `13 1` / `f13_1` | 約 1.46 MiB | 約 186.87 MiB |
| `17 1` / `nth_prime17_inc` | 約 28.07 MiB | 約 3,595.70 MiB |

這是配置公式的結果，並不是實測 GPU 用量或完成時間。20-bit 的需求還取決於 cycle 數量及每一步 Hamming distance。CUDA packed 格式支援 1–27 個 state bits，且單候選解未約簡閘數上界須不超過 `2^27−1`；原 int offset 與機率索引也有 checked capacity。超過限制會報錯，實際可執行規模仍由 RAM／VRAM 與時間決定。

## 5. 實機逐代驗證

`reference_cpp/` 是原始 C++ sources（與 `../cpp/src` 相同，已同步移除 ROT 系列），作為逐代比對的基準，可獨立編譯。從專案根目錄：

```bash
cmake -S reference_cpp -B build-reference -DCMAKE_BUILD_TYPE=Release
cmake --build build-reference --config Release --parallel
python3 tests/regression.py --baseline build-reference/qbcs --candidate build/qbcs_cuda --backends cuda --gpu-decodes thread,warp --work-dir regression_cuda
```

Windows 請將兩個執行檔路徑改成 `build-reference/Release/qbcs.exe` 與 `build/Release/qbcs_cuda.exe`，`python3` 可改為 `python`。

完整 regression 涵蓋全部 10 種演算法、奇偶 N、delta 邊界、identity、distance-1、不同 thread 數與跨 batch 拆分。每次比較所有 trial 的完整歷史、各 trial 最佳閘數與全域最佳電路的完整 gate list；時間欄位除外。詳細操作見 `tests/README.md`。

GPU 上可另以 Compute Sanitizer 檢查小問題。以下命令不是本次環境的已通過項目：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/qbcs_cuda 5 1 1 --backend cuda --gpu-decode thread --experiments 3 --iterations 5 --threads 3 --seed 2026 --check-gpu --verify --out sanitizer_thread
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/qbcs_cuda 5 1 1 --backend cuda --gpu-decode warp --experiments 3 --iterations 5 --threads 3 --seed 2026 --check-gpu --verify --out sanitizer_warp
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/qbcs_cuda 5 1 1 --backend cuda --gpu-decode warp --experiments 3 --iterations 5 --threads 3 --seed 2026 --check-gpu --out sanitizer_sync
```

## 6. 自動找較快配置

在 GPU correctness 檢查通過後，執行：

```bash
python3 tools/benchmark.py --binary build/qbcs_cuda --baseline build-reference/qbcs --bits 13 --problem 1 --algorithm 1 --experiments 32 --iterations 200 --neighbors 10 --threads 1,4,8,16 --gpu-batches 32,128,512 --gpu-decodes thread,warp --out benchmark_13bit
```

每種配置先 warm up，再量測 3 次完整 process wall time，報告中位數。每次都以相同 seed 比對完整結果，通過後才列出加速比，同時比較相同 thread 數 CPU 及所有已測 CPU 設定中的最快者。輸出 `benchmark_summary.csv`、`benchmark.json` 與完整 run logs。`--out` 請使用新資料夾。

第一次可先縮小為 `--threads 4,8 --gpu-batches 32,128 --repeats 1 --warmups 0`，再針對較好的區域作正式重複量測。GPU 摘要會列出實際 batch 大小與 H2D+clear／decode／reduce／D2H 各階段時間，便於判斷下一個瓶頸。

原始 C++ 在一個 13-bit AE-QTS instrumented profiling 中，合成約占 sampled CPU time 的 58.4%，抽樣／修復約 26.3%，更新約 15.2%。這說明保留 CPU 抽樣與更新會留下可觀成本；本版本的真實加速幅度需要完整 wall-time 實測。請勿將 CUDA core 數量、編譯器 register 數據或 emulated 測試時間換算成加速倍數。完整 profile 與限制在 `validation/original_cpu_profile.txt`。

## 7. 多張 GPU／多台電腦

每個程序使用一張 GPU。不同程序用相同 base seed、互不重疊的 trial offset，即可平行執行同一研究設定。例如將 100 trials 分為 40／30／30：

```bash
./build/qbcs_cuda 13 1 1 --experiments 40 --trial-offset 0 --seed 2026 --gpu-device 0 --out part_000_039
./build/qbcs_cuda 13 1 1 --experiments 30 --trial-offset 40 --seed 2026 --gpu-device 1 --out part_040_069
./build/qbcs_cuda 13 1 1 --experiments 30 --trial-offset 70 --seed 2026 --gpu-device 2 --out part_070_099
```

以上三個程序可在不同終端同時啟動。在不同電腦上，各自使用當地的 GPU index（例如都為 0），並保持其餘實驗參數完全相同。分割方式不合併 VRAM，也不需要 trial 之間交換搜尋狀態。CSV 的 trial 編號使用全域編號；每個分片自己的平均／標準差仍只代表該分片，彙整整體實驗時須使用全部 trial rows 重新計算。

`tests/shard_regression.py` 已验证 trial seed／分片／合併結果一致性；正式實驗可沿用相同切分方式。

## 8. 專案索引

| 路徑 | 內容 |
|---|---|
| `src/cuda_evaluator.cu` | 真正 CUDA kernels、pinned buffers、VRAM 配置、階段計時 |
| `src/cuda_core.h` | CPU／CUDA 共用的精確整數 decoder 與 reducer |
| `src/batch_layout.*` | topology、L1／L2 schedule、容量估算與打包 |
| `src/evaluation.*` | 多 trial 批次佇列、分批、回傳與錯誤傳播 |
| `src/quantum_algos.cpp`、`src/baseline_algos.cpp` | 原搜尋邏輯與安全的 batch 邊界 |
| `src/synth.cpp` | 原 CPU 合成器／最後最佳電路輸出／驗證參照 |
| `reference_cpp/` | 原 C++ source（與 `../cpp/src` 相同），regression 的比對基準 |
| `tests/` | 共用核心、排程、逐代對照、Python 重播、分片驗證 |
| `tools/benchmark.py` | CPU／CUDA 配置掃描、精確結果比對與完整 wall time |
| `docs/` | 設計推導、兩版來源差異、驗證證據 |
| `validation/`、`tests/validation/` | 編譯、CPU profiling、參照 hashes 與測試報告 |

授權保留原 `LICENCE`：MIT，2026 Wei-Chieh Lai and Kuo-Chun Tseng, ASC Lab, National Ilan University。
