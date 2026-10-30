# 參照版本與語義稽核

本紀錄依使用者提供的 Python 與 C++ 壓縮檔，以及本次實際執行的重播測試撰寫。原始程式碼保留在工作用參照目錄；以下 `Python/` 與 `C++/` 分別表示兩個原始專案的根目錄。行號指原始上傳版本。

## 移植的判定基準

CUDA 版採用上傳 C++ 版的演算法與抽樣行為作為執行基準，Python 版用來交叉確認四層編碼、合成電路、fitness 排序與機率更新。兩個原始版本的亂數產生器及亂數消耗順序已不同，因此原本就不保證相同 seed 產生相同的完整搜尋軌跡。

應分開檢查以下三件事：

1. **同一組四層 sample 的決定性結果**：電路 gate list 與順序、簡化後 gate count 必須相同。只有功能等價、gate count 卻不同，仍不足以符合移植要求。
2. **同一 C++ seed 的搜尋結果**：保留 RNG 呼叫順序、fitness 排序、tie 處理與更新運算，使歷代 best gate count 與 best sample 不因批次計算而改變。
3. **電路功能**：輸出電路須實現輸入真值表。這個檢查補充前兩者，不能單獨用來證明搜尋演算法等價。

## Python → 原 C++ 實際重播結果

本次使用上傳 C++ 專案既有的 `tests/dump_replay.py` 與 `tests/replay_test.cpp`，以原 Python 專案產生 sample，再交由原 C++ 版本重播。所有測試的族群數均為 10、seed 均為 123、problem index 均為 1。

| 題目 | 演算法 | 代數 | 相同電路數 | 已比對 gate 數 | 結果 |
|---|---|---:|---:|---:|---|
| 3 bit | AE-QTS | 100 | 1,000 | 18,440 | 通過 |
| 5 bit | QTS | 30 | 300 | 37,250 | 通過 |
| 5 bit | QEA | 30 | 300 | 37,312 | 通過 |
| 8 bit | AE-QTS | 3 | 30 | 49,838 | 通過 |
| **合計** | | | **1,630** | **142,840** | **全部通過** |

每一組測試均確認：

- 同一 sample 合成的 gate list，包括 gate 順序與每個控制／目標位元，完全相同。
- 每個合成電路均通過完整真值表模擬。
- 依 fitness 排序的 sample index 完全相同。
- 每一代更新後的四層 double 機率陣列相同。

**測試範圍限制：**這些是 Python 對原 C++ 的固定 sample 重播結果，並非 GPU 實機結果。既有 `replay_test.cpp` 內另寫了一份機率更新函式供重播，不是直接呼叫完整演算法 driver；因此還需要 CUDA 專案的 driver regression 與 GPU differential test。本表不代表所有演算法的 Python/C++ 全部流程均經 bitwise 驗證。

重播所依據的程式位置：`C++/tests/dump_replay.py:70–101`、`C++/tests/replay_test.cpp:79–125`。

## 四層編碼與合成規格

| 步驟 | 必須保留的行為 | 原始程式位置 |
|---|---|---|
| Cycle decomposition | 依輸入 index 由小至大走訪 permutation；排除 fixed points；每個 cycle 先反轉再首尾閉合 | `Python/utils/init_state.py:6–36, 83–107`；`C++/src/encoding.cpp:95–150` |
| Layer 1 | 每個 cycle 編碼為 MSB-first binary priority，依 priority **穩定降冪**排序；同分保留原 cycle index 順序；只有一個 cycle 時仍保留一個 bit | `Python/utils/init_state.py:51–60`；`Python/utils/topology.py:35–37, 90–92`；`C++/src/synth.cpp:139–150` |
| Layer 2 | 使用 `ceil(log2(cycle_length))` bits 解碼進入點，再對 mapping 取模；2-cycle 的 mapping 為 1，其 bit 仍存在但解碼結果恆為 0 | `Python/utils/init_state.py:83–96`；`Python/utils/topology.py:39–45`；`C++/src/encoding.cpp:136–148` |
| 選取 transition | 一般長度為 L 的 cycle，由進入點環狀依序取 L−1 個 transition；2-cycle 只取一個 transition | `Python/utils/topology.py:98–120`；`C++/src/synth.cpp:152–165` |
| Layer 3 | Hamming distance 為 d 的 step 具有 d 組 priority，每組 `ceil(log2(d))` bits；先穩定降冪排序 priority，再按照 **MSB-first** 的差異位元位置依序 flip | `Python/utils/topology.py:587–626`；`C++/src/synth.cpp:75–109` |
| Layer 4 | d > 1 時保留 2d−2 個 head/tail decisions；修復後 0、1 各 d−1 個；依前半、中央 pivot、交換 pointers、後半的順序輸出 2d−1 個 raw gates | `Python/utils/init_state.py:353–392`；`Python/utils/topology.py:723–757`；`C++/src/synth.cpp:112–131` |
| d = 1 | 直接輸出一個 gate；Python 的 `999` sentinel 在 C++ 中改由零長度 Layer 3/4 區間表示 | `C++/src/encoding.cpp:159–175`；`C++/src/synth.cpp:70–72` |
| Gate 表示 | 每個 gate 恰為一條 hypercube edge，除一個 target 外，其餘 qubit 都有 0 或 1 控制；target 的整數 bit index 以 LSB 為 0，輸出 list 以 MSB 為先 | `C++/src/qbcs.h:127–135`；`C++/src/synth.cpp:190–203` |
| Fitness | 整條電路通過 Deletion Rule／Moving Rule 後仍存活的 gate 數，越小越好；沒有其他加權項 | `Python/utils/topology.py:65–67, 759–792`；`C++/src/synth.cpp:175–186` |

`Python/utils/topology.py:121–163` 計算的 adjacency continuity 資料未參與回傳值或 fitness。該檔案的 `reduce_sequences_*`、`determine_transition_strategy` 等函式也不在目前呼叫路徑內。移植不應重新啟用這些舊函式，也不應加入路徑 heuristic。

### Moving Rule 與消去的順序限制

累積器記錄每個頂點最近一個仍存活的 gate。對新 edge 的兩個端點，取兩個頂點堆疊中 index 較大的 gate；若其 target 與新 gate 一致，即為相同 edge，可成對消去，並回復兩個端點的 previous links。否則新 gate 加入堆疊。

依據：`Python/utils/topology.py:628–702`、`C++/src/synth.cpp:33–60`。

這個狀態具有順序依賴，且消去可能跨越 transition 與 cycle 邊界。可以平行準備各 step 的 raw gates，但送入每個 sample 的累積器時，必須保留原本的完整 emission order。不能分別簡化每個 cycle、把各 cycle 的 gate counts 相加，再當成原本 fitness；也不能只找上一個同名 gate 而忽略中間觸及端點的 gate。

## 抽樣、更新與相同 seed 的風險

### 原 Python/C++ 已存在的差異

- Python 使用 `random`，部分 baseline 同時使用 NumPy RNG；C++ 使用 xoshiro256++，以 splitmix64 初始化。兩者的同 seed 不對應同一亂數流。
- Python `gen_nbrs` 以 **layer → population** 的順序抽样；C++ driver 以 **individual → layer** 的順序抽樣。參見 `Python/utils/init_state.py:180–198`、`C++/src/quantum_algos.cpp:160–166`。
- Python Layer 4 每一個 step 抽完立即 repair；C++ `sample_individual` 先抽完所有 Layer 4 bits，再逐 step repair。Python 的 d = 1 Layer 4 sentinel 也會消耗一次無效 threshold；C++ 沒有這個 bit。參見 `Python/utils/init_state.py:321–344`、`C++/src/encoding.cpp:242–250`。
- Python AE-QTS、QTS、QEA 在主要迴圈前會額外抽出整代並合成後丟棄；C++ 未保留此段。參見 `Python/core/AE_QTS.py:31–36`、`QTS.py:32–37`、`QEA.py:38–43`。
- C++ 已修正原 Python 部分演算法回傳 best circuit 的包裝／舊資料問題，且輸出改用 CSV／UTF-8。原始 C++ README 的「與 Python 版有意為之的差異」第 1–6 點亦有說明。

以上不代表 CUDA 可以任意更動 C++ RNG 順序；本次移植的 C++ 相容路徑應保留 C++ 既有行為。

### 必須保留的細節

- **Layer 1–3 抽樣使用 `P[0] < r` 才選 1；Layer 4 使用 `P[0] > r` 才選 0。**兩者在 equality 時行為不同，不能統一改写比較符號。`r` 來自 [0,1) 的 53-bit double。
- Layer 4 repair 在多數 bit 中均勻、不重複地選取位置翻轉，選取數為零一數量差的一半。不能改成固定位置修復或重新獨立抽樣。
- Fitness 使用穩定升冪排序，同 gate count 保留 population index 順序。即使一對個體 fitness 相同，原程式仍會在 decision bits 不同時更新；不能自行跳過這些 pairs。
- Global best 只有 **嚴格改善 `<`** 才替換，不能把等分的新 sample 蓋掉舊 sample。
- AE-QTS 每一個 bit 的 pairs 必須按照 t = 0 到 `floor(N/2)−1` 順序更新；每一 pair 使用 `delta/(t+1)`，且 **每一次 pair 更新後立即進行 boundary correction**。不能先將 harmonic updates 加總，再只 clamp 一次。
- 四層中不同 decision bits 可以平行，但同一 bit 的 pair 更新順序必須保留。對 P[0]、P[1] 仍應保留兩個 double 及既有加減順序；直接換成單一 probability、float 或 fast math 可能改變軌跡。
- 固定迭代上限、歷代 best 記錄位置及每代 sample 數應保留。移植不加入 entropy 停止、restart、adaptive population 或額外局部搜尋。

依據：`C++/src/encoding.cpp:19–56, 209–250`；`C++/src/algorithms_common.cpp:34–37`；`C++/src/quantum_algos.cpp:82–111, 160–225`。

### 名稱相似但規則不同的量子啟發式演算法

| ID | 名稱 | 實際更新規則 |
|---:|---|---|
| 1 | AE-QTS | 當代第 t 名 best 對第 t 名 worst；機率直接加減 `delta/(t+1)` |
| 2 | QTS | 當代 single best 對 single worst；機率直接加減 delta |
| 3 | QEA | 每個 bit 都朝 historical global best 加減 delta |
| 10 | AE-QTS-GB | 每一 pair 的 best 都使用 historical global best；機率直接加減 `delta/(t+1)` |

ID 1、2、3、10 雖然使用 `delta_theta` 變數名稱，實際上不是振幅旋轉。應以程式運算為準，不把研究文件中的角度式自動代入機率版本。原 C++ 另有 ID 11–13 的振幅旋轉（ROT）系列，已從本專案移除。

## 本次 fitness batching 的安全邊界

`evaluate` 不消耗 RNG。對原本互不依賴的 candidates，先保留原順序產生全部 samples，再批次計算 fitness，可以保留抽樣與更新行為。本次只在下列位置改用 `evaluate_many`。

| 演算法 | 批次位置 | 保留的相依關係 |
|---|---|---|
| 所有 quantum IDs | 每代全部 N 個 samples | 先完成全部 fitness，再穩定排序、更新 global best 與機率 |
| GA | 初始族群、每代 N 個 children | tournament 仍使用同一舊族群；保留 elite 加入 N+1 後的穩定截選 |
| DE | 初始族群、每代全部 trial samples | 使用額外 trial buffer；全部 trial 由同一舊族群產生，再依原本 `<=` 條件接受並同步替換 |
| TS | 每代 N 個候選 | 先形成全部 move，再依排序執行 tabu／aspiration 判斷 |
| PSO | 初始族群、每代全部粒子 | 保留當代 position/velocity 計算，fitness 完成後才更新 pbest／gbest |
| WOA | 初始族群 | 每代 candidate 的產生、接受、population／leader 更新仍逐一進行 |
| ABC | 初始族群 | 保留 scout → employed → onlooker 的原地更新順序與預算 |

WOA 在每個 candidate 接受後立刻更新 population，若改善 best 也立刻更新 leader；之後的 candidate 會讀到這些新資料。ABC 的 employed 與 onlooker 也會讀到先前更新的 population；onlooker 的輪盤權重則在 phase 開始前計算一次。這兩個演算法若將整代候選全部從舊族群生成，再統一接受，就改變了原演算法。依據：`C++/src/baseline_algos.cpp:410–431, 479–513`。

## 容量與實作檢查

- Raw gate buffer 的容量必須容納所有被選 transition 的 `2d−1` gates，不能以最終縮減 gate count 配置。
- Raw index／previous links 必須在每個 sample 自成一套，不能跨 samples 或跨 trials 共用狀態。
- 每次 sample 完成後需清除其頂點 heads；原 C++ 只清除碰觸過的端點。CUDA 使用整塊清零或其他等價方式時，也必須避免遺留前次 sample 的 heads。
- 大問題的容量計算、offset 乘法與分配大小應用足夠位寬並檢查溢位；不要僅因 gate target 可用 32 bits 表示，就宣称實際記憶體可支援 32-qubit 真值表。
- 驗證 GPU 核心时，至少需要同 samples 的 CPU/GPU gate count 比對；若核心提供 gate list，亦應比對 emission order。只有統計平均近似不能排除順序與 memory corruption 問題。

以上限制服務於「整體邏輯與目標不改變」的要求；效能改善應来自批次、資料配置、併行解碼與等價的程式結構重排。
