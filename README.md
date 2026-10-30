# A Stepwise Encoding Framework for Optimization of Quantum Boolean Circuit Synthesis

本專案以**四層逐步編碼（Stepwise Encoding）**表示量子布林電路合成的可行解空間，並以 AE-QTS 等元啟發式演算法搜尋 gate 數最少的電路。

研究最初以 **Python** 實作。為了**提高執行速度**並**處理更高位元數的題目**，我們使用 **Claude AI** 協助將 Python 版改寫為 **C++** 與 **CUDA** 版本。三個版本使用相同的編碼、電路合成規則與 fitness（約簡後的 gate 總數），並經過逐一比對驗證（見[一致性驗證](#一致性驗證)）。

## 三個版本

| 資料夾 | 語言 | 用途 | 演算法 |
|---|---|---|---|
| [`python/`](python/) | Python 3 | 原始研究實作，方便閱讀與修改 | ID 1–9 |
| [`cpp/`](cpp/) | C++17 | 單執行緒約比 Python 快 120–180 倍，支援多執行緒平行跑 trial | ID 1–10 |
| [`cuda/`](cuda/) | C++17 + CUDA | 搜尋流程仍在 CPU 執行，電路合成與 fitness 計算移到 GPU 批次處理，適合 13-bit 以上的題目 | ID 1–10 |

| ID | 演算法 | ID | 演算法 |
|---:|---|---:|---|
| 1 | AE-QTS | 6 | TS |
| 2 | QTS | 7 | PSO |
| 3 | QEA | 8 | WOA |
| 4 | GA | 9 | ABC |
| 5 | DE | 10 | AE-QTS-GB（僅 C++／CUDA） |

C++／CUDA 的命令列參數為 `[位元數] [題號] [演算法 ID]`，省略時會像 Python 版一樣逐一詢問。題庫放在各自的 `data/problems.txt`，內容相同。

## 快速開始

### Python

```bash
cd python
pip install numpy pandas openpyxl
python main.py            # 依提示輸入位元數、題號、演算法
```

### C++

需要 C++17 編譯器與 CMake（或直接用 g++）。

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/qbcs 5 1 1 --experiments 10 --iterations 1000 --seed 2026
```

完整選項見 [`cpp/README.md`](cpp/README.md)。

## CUDA 版部署

### 1. 環境需求

| 項目 | 需求 |
|---|---|
| 顯示卡 | NVIDIA GPU，並安裝最新驅動程式 |
| CUDA Toolkit | 12.x 以上；RTX 50 系列需 12.8 以上 |
| CMake | 3.24 以上 |
| 編譯器 | Windows：Visual Studio 2022（勾選「使用 C++ 的桌面開發」）／Linux：CUDA Toolkit 支援的 GCC 版本 |

確認安裝成功：

```bash
nvidia-smi        # 看得到顯示卡與驅動版本
nvcc --version    # 看得到 CUDA Toolkit 版本
```

### 2. 查詢顯示卡架構代號

編譯時要指定 `CMAKE_CUDA_ARCHITECTURES`。常見顯示卡：

| 顯示卡 | 代號 |
|---|---|
| RTX 30 系列（如 3080、3090） | `86` |
| RTX 40 系列（如 4080、4090） | `89` |
| RTX 50 系列（如 5080、5090） | `120` |

其他型號請查 [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus)。

### 3. 編譯

**Windows**（在「x64 Native Tools Command Prompt for VS 2022」中執行；以下以 RTX 40 系列為例）：

```powershell
cd cuda
cmake -S . -B build -A x64 -DQBCS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release --parallel
```

**Linux／WSL**：

```bash
cd cuda
cmake -S . -B build -DQBCS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 4. 先驗證 GPU 結果正確

第一次在新機器上執行時，請先跑小題目並加上 `--check-gpu`（每個 GPU 算出的 gate 數都和 CPU 比對，不一致會立即報錯）與 `--verify`（驗證每個電路的真值表）：

```powershell
# Windows
.\build\Release\qbcs_cuda.exe 5 1 1 --backend cuda --experiments 3 --iterations 20 --threads 3 --seed 2026 --check-gpu --verify --out exp_check
```

```bash
# Linux
./build/qbcs_cuda 5 1 1 --backend cuda --experiments 3 --iterations 20 --threads 3 --seed 2026 --check-gpu --verify --out exp_check
```

### 5. 正式執行

確認無誤後拿掉 `--check-gpu` 與 `--verify`，例如 13-bit 第 1 題、AE-QTS：

```bash
./build/qbcs_cuda 13 1 1 --backend cuda --experiments 100 --iterations 10000 --threads 8 --gpu-batch 128 --seed 2026 --out exp_cuda
```

結果輸出在 `--out` 指定的資料夾（`.txt` 摘要與 `.csv` 收斂曲線）。常用 GPU 參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--backend` | `cuda` | `cuda`、`cpu`（不用 GPU）或 `emulated`（在 CPU 上模擬 GPU 核心，除錯用） |
| `--gpu-batch` | 128 | 每批送進 GPU 的候選解數量上限 |
| `--gpu-device` | 0 | 有多張顯示卡時指定使用哪一張 |
| `--gpu-memory-mb` | 0 | 顯示記憶體上限（MiB）；0 = 自動使用約 80% 可用量 |

沒有 NVIDIA 顯示卡時，可用 `-DQBCS_ENABLE_CUDA=OFF` 編譯，並以 `--backend cpu` 執行。架構細節、記憶體估算、多 GPU 分工與效能測試見 [`cuda/README.md`](cuda/README.md)。

## 一致性驗證

- **C++ 與 CUDA**：相同 `--seed` 下，每個 trial 的逐代收斂歷史、最終 gate 數與最佳電路完全相同。回歸測試涵蓋 ID 1–10、9 組測試案例、1 與 3 執行緒，以及 CPU 與模擬 GPU 兩種後端（[`cuda/tests/regression.py`](cuda/tests/regression.py)）。
- **Python 與 C++**：以原 Python 程式抽出的 sample 交給 C++ 重算，合成的 gate list（含順序）、fitness 排序與每代機率更新皆完全相同（[`cuda/tests/python_replay.py`](cuda/tests/python_replay.py)）。
- **Python 與 C++ 的亂數產生器不同**（Python `random`／NumPy，C++ xoshiro256++），因此相同 seed 不會得到逐代相同的結果，只在統計上一致。

## 已知限制

- CUDA 版已在 CPU 模擬模式下通過所有等價測試，**尚未在實體 NVIDIA GPU 上完成驗證**。請依上方第 4 步以 `--check-gpu` 確認。
- 以 MinGW（g++）在 Windows 編譯的 C++／CUDA-CPU 版，在極少數題目以多執行緒執行時偶爾會發生 heap corruption 而中止。原始 C++ 版也有此現象；遇到時可改用 `--threads 1`，或以 Visual Studio 編譯。

## 授權

MIT License，見 [`LICENCE`](LICENCE)。Copyright (c) 2026 Wei-Chieh Lai and Kuo-Chun Tseng, ASC Lab, National Ilan University.
