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

## 📐 物理模型架構 (Physics Model Architecture)

本專案利用 PINN 求解控制化學反應速率的常微分方程式 (ODE)。模型的核心目標是根據實驗數據，反推不同溫度/壓力下的反應速率常數與動力學機制。

### 1. 控制方程式 (Governing Equation)

我們考慮固態反應動力學的一般形式：

$$
\frac{d\alpha}{dt} = k(T) \cdot f(\alpha)
$$

其中：
* $\alpha$：反應轉化率 (Conversion fraction, $0 \le \alpha \le 1$)。
* $t$：時間 (Time)。
* $k(T)$：反應速率常數 (Rate constant)，通常隨溫度 $T$ 變化。
* $f(\alpha)$：反應機理函數 (Reaction model function)，例如 JMAK 模型、收縮體積模型等。

### 2. 神經網路架構 (Network Architecture)

我們建構一個全連接神經網路 (FCNN) 來逼近解 $u(t) \approx \alpha(t)$：
* **Input**: 時間 $t$ (以及可選的溫度 $T$ 或壓力 $P$)。
* **Output**: 預測的轉化率 $\hat{\alpha}(t)$。
* **Parameters**: 網路權重 $W, b$ 以及待反演的物理參數 (如 $k$)。

### 3. 損失函數設計 (Loss Function)

為了訓練模型並同時求解反問題，我們定義總損失函數為：

$$
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}
$$

* **數據誤差 (Data Loss)**：
    確保網路預測值符合實驗觀測數據。
    $$
    \mathcal{L}_{data} = \frac{1}{N_{data}} \sum_{i=1}^{N_{data}} \left( \hat{\alpha}(t_i) - \alpha_{exp}^{(i)} \right)^2
    $$

* **物理殘差 (Physics/Residual Loss)**：
    利用自動微分 (Automatic Differentiation) 計算 $\frac{d\hat{\alpha}}{dt}$，強迫網路輸出滿足上述 ODE。
    $$
    \mathcal{L}_{physics} = \frac{1}{N_{phys}} \sum_{j=1}^{N_{phys}} \left( \frac{d\hat{\alpha}}{dt}\bigg|_{t_j} - k \cdot f(\hat{\alpha}(t_j)) \right)^2
    $$
