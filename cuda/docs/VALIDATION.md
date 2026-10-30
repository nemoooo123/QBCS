# 驗證紀錄與尚未量測的部分

日期：2026-09-23。來源為本次上傳的 Python archive 與 `qbcs_cpp.zip`。來源 archives／未修改 C++ 參照 hashes 見 `validation/source_manifest.json`。

## 已完成的驗證

| 類型 | 實際結果 | 能支持的結論 |
|---|---|---|
| 真正 CUDA 編譯 | NVCC 13.0.88、GNU C++ 13.3、`sm_120`；完整 compile/link 通過 | kernels 與 host code 可由真正 CUDA 編譯器建構 |
| CPU-only build | `QBCS_ENABLE_CUDA=OFF`；Release build／CTest 通過 | 無 CUDA 也可編譯參照與共用核心驗證工具 |
| Python 固定樣本重播 | 4 組 traces，1,630 circuits／142,840 gates | 相同編碼時的 gate list／順序、fitness sort 與 replay 機率更新一致 |
| 原 C++ → 新 CPU／emulated | 468 組完整 CLI 比對，0 failures | 13 個 production algorithms 的每 trial 每代輸出及全域最佳 gate list 一致 |
| batch／wait 邊界 | 84 組額外 CLI 比對，0 failures | capacity 小於 population、不同等待與分批策略不改結果 |
| Trial 分片 | 全 13 演算法，6 trials 拆成 2+1+3 後一致 | `--trial-offset` 保留相同 seed／trial 對應 |
| 共用整數核心 | 42,978 samples／40,423 batches／1,350,216 raw gates | shared scalar decoder／reducer 與原完整電路一致 |
| 排程服務 | 24 threads、2,400 requests／45,472 samples；故障注入通過 | concurrent merge／split 的順序、容量與例外傳播正確 |
| Host 記憶體檢查 | core 與 service 的 ASan + UBSan 通過 | 已測 host 路徑無檢出的越界或 undefined behavior |
| Benchmark 控制工具 | 6 次 CPU／emulated smoke runs 結果一致 | 計時、讀取／比對與報告生成工具可執行 |

552 組 CLI 比對中的每次 correctness run 都使用 `--verify`，emulated 亦使用 `--check-gpu`。完整 JSON 結果在 `tests/validation/`。時間欄位不列入相等判斷；每 trial 完整 history、平均／標準差、最終 gate count 與報告中的完整 global best gate list 都有比對。原 CLI 未輸出每個 trial 的獨立最佳電路，故此處沒有宣稱比對未輸出的 gate lists。

## 共用核心測試的獨立參照

`tests/core_test.cpp` 使用上傳版本的 `Synthesizer` 當作一個參照；另用朴素 reverse scan，只允許互不相交 edges 交換再抵消相同 gates，重建完整 gate list。後者不使用 endpoint stacks／packed tokens／predecessor arrays，因此不只是重寫被測 reducer 的同一段實作。完整電路另以原 truth table 驗證。

範圍包含：

- 全部 40,320 個 3-bit permutations，含 identity。
- 3／4／5／8-bit 的 randomized multisample batches。
- Batch stride 1、2、7、17、33、35、384，並重用 layout 調整 batch。
- L1／L3 全零與全一 ties，L2 modulo，distance-1 與問題最大 distance。
- Distance-3 的 64 種 L3 × 6 種合法 L4，共 384 組完整組合。
- 反序執行 step／lane decoder，檢查平行工作互不相依。
- 實際命中 37,486 次 gate cancellation，其中 15,114 次非相鄰、3,063 次跨 cycle。
- Buffer guards 及每個 sample 的有效 emission 尾端檢查。

Core 與 service 的 AddressSanitizer／UndefinedBehaviorSanitizer 都通過。環境的 ptrace 限制使 LeakSanitizer 無法啟動，因此執行時設 `detect_leaks=0`；本紀錄沒有宣稱 leak checking 通過。

## Warp decoder review

Warp kernel 已被真正 NVCC 編譯。另做唯讀 source review 與獨立 host 公式比對：9,825 組 cases、250,773 gates，涵蓋 d=1..27、d<=7 的全部 balanced L4、較大 d 的隨機／tie priorities。Route 與 head／tail emission index 均和原 pointer 方法一致。

以上 host 公式檢查不會執行 CUDA shuffle／ballot 指令，不能用來取代 GPU 實機驗證。提供 `--gpu-decode thread|warp` 讓 GPU regression 能逐一比對兩個真實 kernel。

## 尚未完成的 GPU 證據

本次環境沒有可用 NVIDIA device／driver。實際執行 `--backend cuda` 時回報 CUDA driver/runtime 不相容、device 不可用，程序以失敗狀態結束。**沒有 GPU kernel execution 成功、Compute Sanitizer 通過或 GPU 加速倍數的實測資料。**

待在 NVIDIA 顯卡電腦執行的項目：

1. `tests/regression.py --backends cuda --gpu-decodes thread,warp`，與未修改原 C++ 比較。
2. `--gpu-batch 1` 與一般 batch 的拆分測試。
3. Compute Sanitizer 的 memcheck／synccheck。
4. `tools/benchmark.py` 的完整 wall-time sweep，和最快已測 CPU 配置比較。

上述工具與命令均已備妥；CUDA 執行失敗或結果不一致時會明確失敗，不改用其他 backend 產生看似成功的結果。Windows 編譯與執行也尚未在此 Linux 環境實測。
