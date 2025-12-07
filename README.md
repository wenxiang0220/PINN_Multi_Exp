# PINN_Multi_Exp: Inverse Solver for Coupled Reaction Kinetics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Stable-red)
![Physics](https://img.shields.io/badge/Physics-Kinetics-green)

**PINN_Multi_Exp** 是一個專為解決複雜化學反應動力學反問題而設計的物理資訊神經網路 (Physics-Informed Neural Network) 框架。

本專案針對鎂合金（AZ61）儲氫材料的吸放氫反應，利用 PINN 結合自動微分技術，同時對多組實驗數據進行擬合，並反推耦合反應中的關鍵動力學參數 ($k_1 \dots k_4$) 與材料常數 ($A_0 \dots E_0$)。

## 📐 數學模型架構 (Mathematical Modeling)

本研究建立了一個包含兩個中間產物相 ($R$ 與 $R_2$) 的耦合常微分方程組 (Coupled ODEs)。模型考慮了反應物濃度 $H$ 與產物之間的化學計量平衡。

### 1. 狀態變數與守恆 (State Variables & Conservation)

我們定義 $H$ 為氣相濃度（如氫氣），其受到兩個固相產物 $R$ 與 $R_2$ 的生成所消耗。根據質量守恆與化學計量比（Stoichiometry）：

$$
H(t) = B_{init} - R(t) - 9R_2(t)
$$

其中 $B_{init}$ 為初始濃度，$R$ 與 $R_2$ 分別代表不同反應階段的生成物。

### 2. 控制方程式 (Governing Equations)

反應速率由以下非線性 ODE 系統描述，這正是 PINN 損失函數中的 **Physics Loss** 核心：

**第一階段反應 ($R$ phase):**

$$
\frac{dR}{dt} = k_1 (A_0 - R) H - k_2 (C_0 + R + 9R_2)
$$

**第二階段反應 ($R_2$ phase):**

$$
\frac{dR_2}{dt} = k_3 (D_0 - R_2) H^9 - k_4 (C_0 + R + 9R_2)^9 (E_0 + 4R_2)^4
$$

其中：
* $k_1, k_3$：正向反應速率常數 (Forward rate constants)。
* $k_2, k_4$：逆向反應速率常數 (Backward rate constants)。
* $A_0, C_0, D_0, E_0$：與材料容量及活性相關的待定參數。

## 🧠 神經網路架構 (Network Architecture)

為了實現多實驗參數共享與精確求解，本框架採用了特殊的網路設計：

1.  **條件輸入 (Conditional Input)**:
    網路輸入包含時間 $t$ 與實驗編號 (Condition ID, `cid`)：
    $$
    \text{Net}(t, \text{cid}) \rightarrow [\hat{R}, \hat{R}_2]
    $$
    這使得單一網路能同時學習不同實驗條件下的動力學曲線。

2.  **硬約束初始條件 (Hard Initial Condition Enforcement)**:
    為了保證 $t=0$ 時產物濃度為零，我們不透過 Loss 懲罰，而是直接在架構上強制滿足：
    $$
    R(t) = t \cdot \sigma(\text{Net}_1(t)) \cdot B_{init}
    $$
    $$
    R_2(t) = t \cdot \sigma(\text{Net}_2(t)) \cdot D_0
    $$
    此方法顯著提升了訓練初期的收斂穩定性。

3.  **參數共享機制 (Parameter Sharing)**:
    反應速率常數 $k_i$ 與材料常數 $A_0 \dots E_0$ 被設計為可訓練的全局變數 (Learnable Variables)，在所有實驗數據間共享，確保物理參數的一致性。

## 🚀 核心演算法 (Algorithm)

程式碼 `inverse_multi_exp.py` 的執行流程如下：

1.  **Data Loading**: 讀取 `AZ61_3Pd` 系列實驗數據，進行時間與濃度的正規化 (Normalization)。
2.  **PINN Training**:
    * 最小化數據誤差：$\mathcal{L}_{data} = ||H_{pred} - H_{meas}||^2$
    * 最小化物理殘差：$\mathcal{L}_{physics} = ||\frac{dR}{dt} - f_1(\dots)||^2 + ||\frac{dR_2}{dt} - f_2(\dots)||^2$
3.  **Verification**: 使用 Runge-Kutta 4 (RK4) 數值方法，代入 PINN 反推的參數進行正向模擬，驗證反演結果的準確性。

## 📂 檔案結構

```plaintext
PINN_Multi_Exp/
├── inverse_multi_exp.py    # [Core] PINN solver & Physics definition
├── dataset/                # Experimental CSV data (AZ61_3Pd...)
├── conf/                   # Configuration via Hydra
├── artifacts/              # Output folder
│   ├── Exp_0_fit.png       # Visualization for Experiment 1
│   ├── Exp_1_fit.png       # Visualization for Experiment 2
│   └── Exp_0_pred.csv      # Numerical results
└── README.md
