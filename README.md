# PINN_Multi_Exp

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Stable-red)
![Status](https://img.shields.io/badge/Status-Active-success)

**PINN_Multi_Exp** 是一個基於物理資訊神經網路 (Physics-Informed Neural Networks, PINNs) 的深度學習框架，專為解決 **多重實驗配置下的反問題 (Inverse Problems)** 而設計。

本專案旨在透過 PINN 同時對多組實驗數據進行擬合，並利用自動微分技術精確反推物理模型中的未知參數（如反應速率常數、擴散係數等）。

## 📂 檔案結構 (File Structure)

```plaintext
PINN_Multi_Exp/
├── inverse_multi_exp.py  # 核心程式：執行 PINN 訓練與參數反推
├── dataset/              # 實驗數據存放區
├── conf/                 # 設定檔 (模型參數、超參數配置)
├── Exp_0_fit.png         # 實驗 0 擬合結果圖
├── Exp_1_fit.png         # 實驗 1 擬合結果圖
└── README.md             # 專案說明文件
'''
🚀 功能特點 (Key Features)
多實驗並行處理 (Multi-Experiment Support) 支援同時讀取多組不同實驗條件（如不同溫度、壓力）的數據，並進行批次訓練。

反問題求解 (Inverse Modeling) 利用 PINN 的 Residual Loss 機制，將物理定律（ODE/PDE）嵌入損失函數中，從數據中反推未知的物理參數。

自動化視覺化 (Automatic Visualization) 訓練過程中自動生成擬合曲線與誤差分析圖（如 Exp_0_fit.png），即時監控模型表現。
