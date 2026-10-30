# qbcs — Quantum Boolean Circuit Synthesis（Stepwise Encoding + AE-QTS）C++17 版

這是 `AEQTS_boolean_circuit`（Python）的 C++17 移植版，功能與命令列介面對齊原本的 `main.py`，
九個演算法（AE-QTS、QTS、QEA、GA、DE、TS、PSO、WOA、ABC）全部移植；
單執行緒約比 Python 快 120～180 倍，另支援多執行緒平行跑 trial。

## 檔案對應

| C++ | 原 Python | 內容 |
|---|---|---|
| `src/qbcs.h` | — | 公用型別／API：`Encoding`、`Sample`、`QLayers`、`Synthesizer`、`Params`、`run_*` |
| `src/encoding.cpp` | `utils/init_state.py` | `find_cycles`、`build_encode`、`gen_nbrs`（四層抽樣）、`repair_sequence_logic`、RNG（xoshiro256++） |
| `src/synth.cpp` | `utils/topology.py` | `decode_and_synthesize` → `synthesize_route` → `generate_state_trajectory` → `assemble_reversible_circuit`（`_MRAccumulator`：A·A = I 消去 + Moving Rule）、`verify_circuit_logic` |
| `src/quantum_algos.cpp` | `core/AE_QTS.py`、`QTS.py`、`QEA.py` | 三者共用主迴圈，只差 `updateQ` 的旋轉規則 |
| `src/baseline_algos.cpp` | `core/GA.py`、`DE.py`、`TS.py`、`PSO.py`、`WOA.py`、`ABC.py` | 六個 baseline |
| `src/data_loader.{h,cpp}` + `data/problems.txt` | `utils/data_loader.py` | 75 個 benchmark truth table（純文字，新增題目不需重編譯） |
| `src/main.cpp` | `main.py` | CLI、多執行緒 trial、輸出 `.txt` / `.csv` |
| `tests/dump_replay.py`、`tests/replay_test.cpp` | — | 與 Python 版的逐位元等價測試（見下） |
| `tools/csv_to_xlsx.py` | — | 選用：把 `.csv` 轉回原本的 `.xlsx` 版面 |

### 資料結構（與 Python 的對照）

Python 以巢狀 list 存四層編碼；C++ 改成 **flat array**，由 `Encoding` 的 offset 表還原巢狀結構：

| 層 | 意義 | Python | C++ |
|---|---|---|---|
| Layer 1 | cycle 優先順序（cycle priority） | `nbr1[i]`：`(num_cycles, bits)` ndarray | `Sample::l1`，`l1_cyc_off[c] + bit` |
| Layer 2 | 斷邊／進入點（break edge / entry point） | `nbr2[i][c]`：`ceil(log2 len)` 個 bit | `Sample::l2`，`l2_off[c] + bit` |
| Layer 3 | 中介節點順序（path nodes） | `nbr3[i][c][s]`：`d` 個節點 × `ceil(log2 d)` bit，或 `[999]` | `Sample::l3`，`l3_off[g] + node*nb + bit` |
| Layer 4 | 閘序（head / tail order） | `nbr4[i][c][s]`：`2d−2` 個 bit，或 `[0]` | `Sample::l4`，`l4_off[g] + bit` |

`999` marker（Hamming distance = 1 的 step）在 C++ 裡**不佔位元**，由 `step_nb[g] == 0` 直接處理，
因此搜尋迴圈內沒有任何動態配置。機率層 `QLayers` 對每個 bit 仍存 `(P[0], P[1])` 兩個 double，
與 Python 的 `(k, 2)` 矩陣完全相同，故 `updateQ` 的浮點運算結果逐位元一致。

## 編譯

需要 C++17 編譯器；沒有任何第三方依賴。

**Linux / macOS / Windows（MSYS2 MinGW-w64 或 WSL）**
```bash
g++ -std=c++17 -O2 -pthread src/*.cpp -o qbcs
```

**Windows（Visual Studio 的 x64 Native Tools 命令列）**
```bat
cl /O2 /std:c++17 /EHsc src\*.cpp /Fe:qbcs.exe
```

**CMake（任何平台）**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

實測 `-O3` 與 `-O2` 速度相同，`-march=native` 反而略慢，故建議 `-O2`。

## 執行

與 `main.py` 相同：`qbcs [num_bits] [problem_idx] [algo_id]`；省略位置參數時會像 Python 一樣互動提問。

```bash
./qbcs 5 1 1                                   # 5-bit 第 1 題，AE-QTS，100 trials × 10000 代
./qbcs 13 2 1 --threads 16 --seed 2026         # 16 條執行緒；固定 seed 可重現
./qbcs 8 3 4 --experiments 30 --iterations 2000
./qbcs 1 --tt "1,0,3,2,5,7,4,6"                # 自訂 truth table（第一個位置參數是 algo_id）
./qbcs 1 --tt-file my_perm.txt
./qbcs 5 1 1 --experiments 3 --iterations 300 --verify   # 每個合成電路都用 2^n 輸入模擬驗證
```

algo_id：1 AE-QTS、2 QTS、3 QEA、4 GA、5 DE、6 TS、7 PSO、8 WOA、9 ABC。

主要選項（預設值 = `main.py`）：

| 選項 | 預設 | 說明 |
|---|---|---|
| `--experiments N` | 100 | trial 數 |
| `--iterations N` | 10000 | 每個 trial 的代數 |
| `--neighbors N` | 10 | 族群／鄰域大小 N |
| `--threads N` | 全部核心 | 平行執行的 trial 數；**結果只由 `--seed` 決定，與執行緒數無關** |
| `--seed S` | 時間 | 基底亂數種子（每個 trial 由 seed 與 trial 編號雜湊出獨立的種子） |
| `--data FILE` | `data/problems.txt` | 題庫檔 |
| `--out DIR` | `exp` | 輸出根目錄 |
| `--verify` | off | 除錯用：模擬每個電路並與 truth table 比對，失敗數會列在報告裡 |
| `--delta`、`--ga-k/--ga-pc/--ga-pm`、`--de-cr`、`--tabu`、`--pso-w/-c1/-c2`、`--woa-b`、`--abc-limit` | 同 Python | 各演算法參數 |

### 輸出

```
exp/<n>_bit/<ALGO>_Results/<ALGO>_<n>_<idx>.txt   # 摘要、最佳電路（Python list 格式）、μ ± σ
exp/<n>_bit/<ALGO>_Results/<ALGO>_<n>_<idx>.csv   # Trial × Generation 矩陣
```

`.csv` 的版面與 Python 用 pandas 寫進 `.xlsx` 的 DataFrame 完全相同：
第一欄是 index（`Trial_1`…、`Average_Convergence`、`Std_Deviation`），欄位 `Gen_1`…`Gen_N`、`Execution_Time(s)`；
`Std_Deviation` 列與 pandas 一樣用樣本標準差（ddof = 1），`.txt` 內的 `μ ± σ` 與 `np.std` 一樣用 ddof = 0。

`plot_results.py` 只需把 `pd.read_excel(path)` 改成 `pd.read_csv(path, index_col=0)`；
或用 `python tools/csv_to_xlsx.py exp` 把整個目錄轉回 `.xlsx`。

### 新增題目

在 `data/problems.txt` 加一行即可（不必重編譯）：

```
<num_bits> <problem_idx> <name> : v0 v1 v2 ... v(2^n-1)
```

## 與 Python 版的等價性驗證

**1. 重播測試（bit-exact）** — `tests/dump_replay.py` 用**原本的 Python 程式**跑 AE-QTS / QTS / QEA，
把每一代抽到的 nbr1–4、Python 合成出的電路、fitness 排序、`updateQ` 後的機率層全部 dump；
`tests/replay_test.cpp` 把同一批樣本餵給 C++ 的 `Synthesizer` 與更新規則，逐位元比對：

```bash
tests/run_replay.sh /path/to/AEQTS_boolean_circuit 5 1 200 aeqts 7
# REPLAY OK: aeqts n=5 idx=1 iters=200 | circuits identical: 2000 (236462 gates), ...
```

已通過：3–10 bit、三個演算法、共約 13,000 個電路 / 2.4 M 個 gate，
gate list（含順序）、排序、機率層全部相同，且每個電路都通過 truth table 模擬。

**2. 六個 baseline** — 亂數序列無法跨語言重現，改以 `--verify` 確認所有電路正確，
並比對收斂曲線（5-bit f1，30 trials × 300 代）：TS Py 85.20 vs C++ 84.27、ABC 98.33 vs 97.67，
其餘七個亦在統計誤差內。

## 效能（單執行緒，AE-QTS，N = 10）

| 問題 | Python | C++ | 倍數 |
|---|---|---|---|
| 5-bit | 6.0 ms/代 | 0.05 ms/代 | ~120× |
| 8-bit | 98 ms/代 | 0.7 ms/代 | ~140× |
| 10-bit | 716 ms/代 | 3.9 ms/代 | ~180× |
| 13-bit | （推估 ~8 s/代） | 47 ms/代 | ~170× |

13-bit 完整實驗（100 trials × 10000 代）：單執行緒約 13 小時，16 執行緒約 1 小時。

## 與 Python 版有意為之的差異

1. `.xlsx` → `.csv`（版面相同，見上）；`.txt` 改為 UTF-8（原為 UTF-16）。
2. 每個 trial 開頭那一次結果被丟棄的 `gen_nbrs + decode_and_synthesize` 沒有保留（只消耗 RNG，不影響結果）。
3. `topology.py` 中從未被呼叫的 `reduce_sequences_*`、`determine_transition_strategy`、`analyze_bit_differences`、
   `map_next_transition_bits`、`initialize_solution_layer`，以及 `synthesize_route` 內算完未使用的
   `redundant_coupling_count`，未移植。
4. 原 Python 兩個不影響 fitness 統計的小 bug 已修正：`DE.py` 回傳的 best circuit 取自初始族群的 stale
   `circuit_solutions`（gate count 正確，電路內容錯）；`QTS/QEA/PSO/WOA` 存的是 `(circuit, count)` tuple。
5. Python 對 `999` marker 的無效操作（TS 翻 Layer 4 的 `[0]`、GA/WOA/ABC 對 marker 突變）在 C++ 是 no-op，
   但抽到 marker 的機率保留，所以搜尋行為的分布不變。
6. 多執行緒：每個 trial 的種子 = hash(`--seed`, trial 編號)，因此同一個 `--seed` 在任何 `--threads` 下結果相同。

## License

MIT，與原專案相同（原作者：Wei-Chieh Lai、Prof. Kuo-Chun Tseng，ASC Lab, National Ilan University）。
