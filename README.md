PINN_Multi_Exp
PINN_Multi_Exp 是一個基於物理資訊神經網路 (Physics-Informed Neural Networks, PINNs) 的深度學習框架，專為解決 多重實驗配置下的反問題 (Inverse Problems with Multiple Experiments) 而設計。

本專案旨在透過 PINN 同時對多組實驗數據進行擬合，並反推物理模型中的未知參數（如反應速率常數、擴散係數等）。

📂 檔案結構 (File Structure)
Plaintext

PINN_Multi_Exp/
├── inverse_multi_exp.py  # 主程式：執行 PINN 訓練與參數反推
├── dataset/              # 存放實驗數據 (例如 .csv 或 .mat 檔)
├── conf/                 # 實驗配置檔 (模型參數、超參數設定)
├── Exp_0_fit.png         # 實驗 0 的擬合結果視覺化
├── Exp_1_fit.png         # 實驗 1 的擬合結果視覺化
└── README.md             # 專案說明文件
🚀 功能特點 (Key Features)
多實驗並行處理 (Multi-Experiment Support)： 能夠同時讀取並處理多個不同條件下的實驗數據（例如不同溫度、壓力或初始濃度），並針對每個實驗訓練對應的網路或共享參數。

反問題求解 (Inverse Modeling)： 利用自動微分 (Automatic Differentiation) 計算物理殘差 (Residual Loss)，精確反推控制方程式中的未知物理參數。

自動化視覺化 (Automatic Visualization)： 訓練過程中自動生成擬合曲線圖（如 Exp_0_fit.png），方便即時監控模型收斂情況。

🛠️ 安裝與使用 (Installation & Usage)
1. 環境設定
建議使用 Python 3.8 以上版本，並安裝以下核心套件：

Bash

pip install torch numpy matplotlib pandas scipy
2. 執行訓練
使用以下指令執行主程式進行參數反推：

Bash

python inverse_multi_exp.py
(如果您的程式支援命令行參數，例如指定 config 檔案，請修改為： python inverse_multi_exp.py --config conf/config.yaml)

📊 結果範例 (Results)
模型會在訓練過程中透過最小化數據誤差 (Data Loss) 與物理誤差 (Physics Loss) 來逼近真實解。

Experiment 0 Fit	Experiment 1 Fit

匯出到試算表

📐 數學原理 (Mathematical Formulation)
本框架求解的一般形式為：

L 
total
​
 =λ 
data
​
 L 
data
​
 +λ 
physics
​
 L 
physics
​
 
其中 L 
physics
​
  為控制方程式（例如 ODE 或 PDE）的殘差：

L 
physics
​
 = 
N 
f
​
 
1
​
  
i=1
∑
N 
f
​
 
​
  

​
  
∂t
∂u
​
 −N(u,λ) 

​
  
2
 
u: 狀態變數 (State variable)

N: 非線性微分算子 (Nonlinear differential operator)

λ: 待求解的物理參數 (Unknown physics parameters)

📝 引用 (Citation)
如果您在研究中使用了此程式碼，請連結至本儲存庫：

程式碼片段

@misc{PINN_Multi_Exp,
  author = {Wen Xiang Zheng},
  title = {PINN_Multi_Exp: Inverse Problems Solver using PINNs},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wenxiang0220/PINN_Multi_Exp}}
}
💡 下一步建議：
補充 requirements.txt：建議您在本地端執行 pip freeze > requirements.txt 並上傳，方便他人安裝環境。

更新數學公式：您可以將上述數學原理中的公式修改為您實際研究的具體方程（例如化學反應動力學方程  
dt
dα
​
 =k(T)f(α)），這會讓您的備審資料看起來更專業。

上傳配置檔說明：如果在 conf/ 資料夾中有 yaml 或 json 檔，可以在 README 中簡單說明如何修改這些參數。
