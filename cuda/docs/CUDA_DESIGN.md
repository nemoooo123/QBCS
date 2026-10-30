# CUDA 設計與等價性說明

## 目標與邊界

目標是在保留既有搜尋過程的條件下，用 GPU 加速完整編碼的電路合成與 fitness 評估。嚴格執行標準為：相同 C++ RNG state、候選解、每一步機率／振幅更新、tie-breaking、接受條件、代數及停止方式。沒有增加路徑 heuristic、降低搜尋維度或更換演算法。

採 CPU／GPU 異質架構，是因為原程式有兩種不同依賴：每條電路的 moving-rule reduction 需要完整先後順序；WOA／ABC 的部分候選解又依賴同代較早的接受結果。CUDA 執行大量獨立合成工作，CPU 保留已經過對照的搜尋與亂數流程。

## 1. 可交換的工作

固定候選解之後，每個被選中的 transposition，其 Layer 3 priority 解碼與 Layer 4 raw gate 生成只讀該 step 的編碼。只要把結果放回原 emission order 的正確位置，生成工作的執行先後不影響輸出。因此可以平行化 `(sample, active_step)`，且不必生成 Layer 2 已省略的 step。

每個 cycle 選擇的 active step 數量固定：2-cycle 為 1，其餘 length L 的 cycle 為 L−1。CPU 的 `BatchLayout::pack` 依原 L1 排序及原 L2 modulo／pointer 邏輯建立 schedule 和 gate offsets。每個 step 精確輸出 `2d−1` 個 gates，讓各 kernel 任務寫入互不重疊的範圍。

各 trial 的 RNG、population、probability state 與 best tracking 完全獨立。排程服務只合併 fitness 請求，並將結果寫回原 caller 順序。當一個 N-size population 被拆成若干 GPU batches 時，caller 等待全部 N 個 fitness 後才繼續，因此 N、generation barrier 與更新順序不變。

## 2. Layer 3 排序的等價轉換

原本的 differing bits 依 MSB 到 LSB 排列；priority 相同時 stable sort 保留這個先後。對 differing bit p 使用 key：

`key = (priority << 5) | p`

將 key 降序排序，在 priority 相同時自然由較高 bit position 優先，正好等價於原 stable tie order。`p<32`，故 priority 與位置不重疊。此處 5 bits 儲存的是 target index。

Thread decoder 先用單一 `path[]` 放 keys，再原地改放按 priority 順序形成的 hypercube edges。因為 Toffoli edge 對交換兩個端點是同一個 gate，Layer 4 只需要選擇 `path[head]` 或 `path[tail−1]`，不必保存每一個 route vertex 與多組 sorting scratch。

Warp decoder 讓各 lane 取得一個 node priority，透過 shuffle 計算 rank；相同 priority 由較小原 index 優先。各 lane 根據 rank 的前綴 XOR 構成 route，再用 ballot 計算 Layer 4 的零位元前綴，得到和序列 head／tail 相同的邊索引。所有 lanes 都參與 shuffle／ballot；warp 相關分支由同一 step 決定。

兩種 decoder 都保留輸出順序。編譯資源報告顯示 thread decoder 有 112-byte local stack，warp decoder 無 local stack；另一方面，warp 對 `[gate][batch]` 的輸出可能有較大 stride。哪個較快必須由實機測量決定。

## 3. Moving Rule 的順序不能移除

原合成器以每個 vertex 的最近 still-live gate 判斷新 gate 能否穿過後續可交換 gates 並抵消。新 gate 讀取兩個端點的 head，選擇較晚的一個；若 target 相同，對應同一 hypercube edge，便抵消並還原兩個 predecessor heads。

此決策依賴前面抵消後的結果，因此本版仍依整條 emission order 逐 gate 執行。每條樣本的 reducer 由一個 GPU lane 擁有，樣本之間平行。若將各 cycle 獨立約簡後直接相加，會漏掉跨 cycle 抵消；若用無序 atomic push，也會破壞 chronological precedence。

### 以 emission ordinal 取代原 raw vector index

原 `raw_` 只有新增 live gate 時 append；本版的 chronological token 以每個 raw emission 的 ordinal 編號，包含當下被抵消而沒有 append 的位置。這兩種編號對「實際 append 的 gates」維持嚴格相同的先後關係。Reducer 只比較先後與循 predecessor link，從不依賴編號連續，因此此轉換保持等價。

Token 定義為：

`token = ((emission_ordinal + 1) << 5) | target`

0 代表空 head。Target 小於 32，所以 `max(token0, token1)` 同時就是較晚的 gate；同一 gate 的两個端點具有相同 token。每個 emission 的兩個 previous tokens 為 scratch arrays 的兩個 uint32 欄位。

若 `max_gates <= 2^27−1`，上述左移不溢位。Packed edge 為 `(ctrl << 5) | target`，要求 state bits 不超過 27。Host 配置時檢查這些限制。最後的最佳 gate list 由原 CPU `Synthesizer` 重建，GPU 日常 fitness 運算不需要 live flags 或將整條最佳電路回傳。

## 4. 記憶體公式

令：

- `V = 2^n`：truth table／hypercube vertex 數。
- `K`：每個候選解的 active step 數，對固定 topology 不變。
- `B34 = l3_size + l4_size`：候選解 L3／L4 的 byte 數。
- `Gmax`：每個候選解的 raw emissions 上界。
- `S`：全部 topology step 數。
- `B`：實際配置的 GPU batch capacity。

對長度超過 2 的 cycle，任一候選解只省略一個 step。該 cycle 的 raw-gate 上界是所有 `2d−1` 加總，扣除其中最小者。對 2-cycle，直接使用唯一 step 成本。各 cycle 上界相加得到 `Gmax`。

主要 device array 每個 sample 的 byte 數為：

`M_sample = B34 + 8K + 12Gmax + 4V + 8`

分别是 input、schedule+offset、packed raw edge+兩個 previous buffers、vertex heads、emitted count+fitness。共享 step metadata 是 `24S` bytes（以本版 Step 結構定義為準）。總 device allocation 約為：

`M_device = B × M_sample + S × sizeof(Step)`

CUDA context、stream/events 與其他 runtime 成本不在上述陣列公式內；auto budget 只採當時 free VRAM 的約 80%，保留餘裕。Pinned host buffers 額外约需 `B × (B34 + 8K + 8)`，而每個並行 trial 在一般 RAM 裡另有其 population／probabilities／最佳解等資料。

較小 batch capacity 只改變執行分批，不改 N 或任何搜尋參數。若連一個 sample 都放不下、索引可能溢位或配置失敗，會停止並清楚回報，沒有截斷 buffers、近似 fitness 或改成抽樣真值表。

## 5. 目前效能上限與量測方法

原 C++ 的一次 instrumented 13-bit AE-QTS profile 把約 58.4% sampled CPU time 歸入 synthesis，26.3% 歸入 sampling／repair，15.2% 歸入 update。分組與完整原始 log 在 `validation/original_cpu_profile.txt`，包含 compiler inlining／gprof 歸因等限制。

若只把此 profile 當作簡化單 trial 模型，且假設原 sampling／update 成本完全固定、GPU synthesis 可瞬間完成、沒有傳輸開銷，Amdahl 的估算上限約為 `1 / (1−0.584) ≈ 2.40`。這只是說明殘餘 CPU 成本，不能當作實際加速的承諾或嚴格上界：跨 trial 重疊、不同 n／N／代數、CPU 飽和程度、資料傳輸與 profiling instrumentation 都會改變比率。

在本版保留相同 C++ seed 軌跡的條件下，最大的主要限制是 CPU 抽樣／修復／更新，以及每個 sample 的序列 reduction。直接更換為 GPU RNG、同時合併 harmonic updates 或改用不同 moving rule，會失去目前精確對照的基礎。若未來考慮進一步的 GPU 常駐抽樣／更新，需要另外設計和驗證 RNG stream 排程、逐 pair clamp 與 repair 順序。

本版提供每 batch 的 H2D+clear／decode／reduce／D2H event timing，另有 CPU packing time 與 batch min／mean／max。`benchmark.py` 以完整程序 wall time 且同 seed 精確比對結果來掃描 threads、batch、decoder，與最快已測 CPU 配置比較。若 GPU 在某個小問題上較慢，也如實報告加速比小於 1。

## 6. 對照檔與可重現性

`reference_cpp/` 是上傳 C++ source 的未修改副本；`validation/source_manifest.json` 記錄原始 archives 與參照檔案 hashes。`src/synth.cpp` 仍作為原始完整電路參照。搜尋 driver 的變更主要是將沒有相依的 evaluate 呼叫拆成 sampling 與批次 evaluation，DE 使用 trial population buffer 延後接受，但所有候選解仍從同一份原 population 產生。

本次驗證到的結果應分別標示：來源重播、CPU production driver 對照、shared-core host execution、service concurrency、NVCC 編譯。這些證據支持移植邏輯與可編譯性；真正 GPU execution 與性能仍須由 GPU regression／Compute Sanitizer／benchmark 補齊。
