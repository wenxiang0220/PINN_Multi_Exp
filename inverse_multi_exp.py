# -*- coding: utf-8 -*-
"""
PhysicsNeMo inverse fit — Multi-Experiment (Shared k, Distinct R/R2)
同時訓練兩組實驗數據，共享反應速率常數 k1..k4 與初始參數 A0..E0
"""

import sympy
import os, time, copy, glob
import numpy as np
import pandas as pd
import torch  # 用於後處理

from sympy import Symbol, Number, Function, exp
from physicsnemo.sym.eq.pde import PDE
import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.constraint import PointwiseConstraint
from physicsnemo.sym.key import Key


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # -------- helpers --------
    EPS = 1e-12
    def p(*a, **k):
        k.setdefault("flush", True)
        print(*a, **k)

    def find_torch_module(obj):
        try:
            import torch as _torch
        except Exception:
            return None
        for name in dir(obj):
            try:
                v = getattr(obj, name)
            except Exception:
                continue
            if isinstance(v, _torch.nn.Module):
                return v
        return None

    def get_core_module(m):
        return m.module if hasattr(m, "module") else m

    def module_device(mod):
        try:
            for p_ in mod.parameters():
                return p_.device
        except Exception:
            pass
        import torch
        return torch.device("cpu")

    # -------- DATA PREPARATION (Multi-Experiment) --------
    base_dir = os.path.dirname(__file__)
    
    # 定義兩份數據的設定：檔名, 起始時間, 結束時間, 實驗ID(cid)
    experiments = [
        ("AZ61_3Pd_4574_8166sec.csv", 4574, 8166, 0.0),
        ("AZ61_3Pd_12738_16334sec.csv", 12738, 16334, 1.0),
    ]
    
    data_list = []
    
    R_u = 8.31446261815324  # kPa·L/(mol·K)

    # 用來儲存每組實驗的 meta data 供後處理使用
    exp_meta = [] 

    for csv_name, t_start, t_end, cid_val in experiments:
        csv_path = os.path.join(base_dir, "dataset", csv_name)
        if not os.path.exists(csv_path):
            p(f"[ERROR] File not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        
        # Slice Data
        df = df[(df["Time_sec"] >= t_start) & (df["Time_sec"] <= t_end)].copy()
        
        # Standardize
        df = df.rename(columns={
            "Time_sec": "Time_sec",
            "PT01_MPa": "pressure",
            "T01[Treac]_℃": "T_reac"
        })
        
        t = df["Time_sec"].to_numpy(dtype=float)
        idx = np.argsort(t)
        t = t[idx]
        P_MPa = df["pressure"].to_numpy(dtype=float)[idx]
        T_C   = df["T_reac"].to_numpy(dtype=float)[idx]
        
        # Calculate Concentration B (H2)
        T_K = T_C + 273.15
        B_meas = (P_MPa + 0.101325)*1e3 / (R_u * T_K) # mol/L
        
        # Normalization
        t0_val, t1_val = float(t.min()), float(t.max())
        DT_val = t1_val - t0_val
        x = (t - t0_val) / DT_val
        B_init_val = float(B_meas[0])
        
        p(f"[DATA] Exp {cid_val}: {csv_name}, rows={len(t)}, DT={DT_val:.1f}s, B_init={B_init_val:.4f}")
        
        exp_meta.append({
            "cid": cid_val, "t_raw": t, "t0": t0_val, "DT": DT_val, 
            "B_init": B_init_val, "B_meas": B_meas, "x": x
        })
        
        # 準備訓練用的 array (N, 1)
        n = len(x)
        data_list.append({
            "x": x[:, None],
            "cid": np.full((n, 1), cid_val),          # Condition ID (0 or 1)
            "B_meas": B_meas[:, None],
            "B_init_in": np.full((n, 1), B_init_val), # 這是變數！
            "DT_in": np.full((n, 1), DT_val)          # 這是變數！
        })

    # 合併所有數據
    x_aug       = np.concatenate([d["x"] for d in data_list])
    cid_aug     = np.concatenate([d["cid"] for d in data_list])
    B_meas_aug  = np.concatenate([d["B_meas"] for d in data_list])
    B_init_aug  = np.concatenate([d["B_init_in"] for d in data_list])
    DT_aug      = np.concatenate([d["DT_in"] for d in data_list])
    
    zeros = np.zeros_like(x_aug)
    ones  = np.ones_like(x_aug)

    # -------- PARAM BOUNDS --------
    K1_LO, K1_HI = Number(1e-5), Number(5e-2)
    K2_LO, K2_HI = Number(1e-7), Number(1e-3)
    K3_LO, K3_HI = Number(1e-7), Number(1e-3)
    K4_LO, K4_HI = Number(1e-9), Number(1e-5)

    A0_LO, A0_HI = Number(1e-8), Number(1.0)
    C0_LO, C0_HI = Number(0.0),  Number(0.0)
    D0_LO, D0_HI = Number(1e-8), Number(1.5)
    E0_LO, E0_HI = Number(0.0),  Number(0.0)

    # -------- PDE (Multi-Exp) --------
    class Multi_Reaction_PDE(PDE):
        def __init__(self):
            # Input variables: x, cid, B_init, DT
            x_var = Symbol("x")
            cid_var = Symbol("cid")
            B_init_in = Symbol("B_init_in")
            DT_in = Symbol("DT_in")
            
            input_variables = {"x": x_var, "cid": cid_var, "B_init_in": B_init_in, "DT_in": DT_in}

            def wrap(val):
                if isinstance(val, str):
                    return Function(val)(*input_variables)
                elif isinstance(val, (float, int)):
                    return Number(val)
                return val

            # 這些現在是變數，不是常數
            Binit_s = Function("B_init_in")(*input_variables)
            DTn = Function("DT_in")(*input_variables)
            
            # --- NN Outputs ---
            # raw_R 和 raw_R2 現在 implicitly 依賴 (x, cid)
            raw_R  = Function("raw_R")(*input_variables)
            raw_R2 = Function("raw_R2")(*input_variables)

            # --- Learnable Initials (A0..E0) ---
            # 這些是全域參數，不應該依賴 x 或 cid (雖然傳入 input_variables 只是定義符號連結)
            A0_raw = Function("A0_raw")(*input_variables)
            C0_raw = Function("C0_raw")(*input_variables)
            D0_raw = Function("D0_raw")(*input_variables)
            E0_raw = Function("E0_raw")(*input_variables)

            sig = lambda z: Number(1)/(Number(1)+exp(-z))

            A0 = A0_LO + (A0_HI - A0_LO) * sig(A0_raw)
            C0 = C0_LO + (C0_HI - C0_LO) * sig(C0_raw)
            D0 = D0_LO + (D0_HI - D0_LO) * sig(D0_raw)
            E0 = E0_LO + (E0_HI - E0_LO) * sig(E0_raw)

            # --- Hard Enforced IC ---
            # R(0) = 0, R2(0) = 0
            # 這是對 normalized x 有效，不管哪一組實驗 x=0 時 R 都是 0
            R  = x_var * (Number(1)/(Number(1)+exp(-raw_R ))) * Binit_s
            R2 = x_var * (Number(1)/(Number(1)+exp(-raw_R2))) * D0

            # Measured B
            B_meas_sym = Function("B_meas")(*input_variables)

            # --- Rate Constants (Global) ---
            k1 = K1_LO + (K1_HI - K1_LO) * sig(Function("k1")(*input_variables))
            k2 = K2_LO + (K2_HI - K2_LO) * sig(Function("k2")(*input_variables))
            k3 = K3_LO + (K3_HI - K3_LO) * sig(Function("k3")(*input_variables))
            k4 = K4_LO + (K4_HI - K4_LO) * sig(Function("k4")(*input_variables))

            # --- Physics ---
            delta = Number(1e-10)
            H_raw = (Binit_s - R - Number(9)*R2)
            H_pos_uncapped = (H_raw + ((H_raw**2) + delta)**(Number(1)/2))/Number(2) + Number(EPS)
            H_pos = sympy.Min(H_pos_uncapped, Number(2.0))
            H_pow = (H_pos ** Number(9.0))

            def pos(z): return (z + (z**2)**(Number(1)/2))/Number(2)
            order_norm = (pos(k2 - k1) + pos(k4 - k3)) / Number(1e-2)

            scale_R, scale_R2, scale_stoich = Number(0.333333), Number(0.3333333), Number(0.3333333333333)

            # Residuals (using variable DTn)
            reaction_R  = R.diff(x_var)  - DTn * (k1*(A0 - R)*H_pos              - k2*(C0 + R + Number(9)*R2))
            reaction_R2 = R2.diff(x_var) - DTn * (k3*(D0 - R2)*(H_pow)           - k4*((C0 + R + Number(9)*R2)**Number(9.0))*((E0 + Number(4)*R2)**Number(4.0)))
            stoich = H_raw - B_meas_sym

            self.equations = {
                "reaction_R_norm":  reaction_R  * scale_R,
                "reaction_R2_norm": reaction_R2 * scale_R2,
                "stoich_norm":      stoich      * scale_stoich,
                "order_norm":       order_norm,
                "R":  R,
                "R2": R2,
            }

    geo = Line1D(0.0, 1.0)
    Multi_Reaction = Multi_Reaction_PDE()

    # -------- NETWORKS (Modified Inputs) --------
    # Main Network: Input (x, cid) -> Output (raw_R, raw_R2)
    # 這讓網絡能根據 cid 學習不同的曲線
    FC = instantiate_arch(
        cfg=cfg.arch.fully_connected,
        input_keys=[Key("x"), Key("cid")],  
        output_keys=[Key("raw_R"), Key("raw_R2")],
    )
    
    # Parameter Networks: Input (c1 / cA) -> Global Constants
    # 這些不依賴 x 或 cid，所以兩組實驗共享同一組 k 和 A0
    rate_constant1 = instantiate_arch(
        cfg=cfg.arch.linear, input_keys=[Key("c1")],
        output_keys=[Key("k1"), Key("k2")],
    )
    rate_constant2 = instantiate_arch(
        cfg=cfg.arch.linear, input_keys=[Key("c2")],
        output_keys=[Key("k3"), Key("k4")],
    )
    param_head = instantiate_arch(
        cfg=cfg.arch.linear, input_keys=[Key("cA")],
        output_keys=[Key("A0_raw"), Key("C0_raw"), Key("D0_raw"), Key("E0_raw")],
    )

    # Init bias tweak
    try:
        import torch
        m = find_torch_module(FC)
        if m is not None:
            core = get_core_module(m)
            last = None
            for mm in core.modules():
                if isinstance(mm, torch.nn.Linear): last = mm
            if last is not None and last.bias is not None and last.bias.numel() >= 2:
                with torch.no_grad():
                    last.bias.fill_(-1.0); last.bias[0] = -6.0
        p("[INIT] Bias set.")
    except Exception as e:
        p("[INIT] Bias tweak skipped:", e)

    nodes = [
        FC.make_node(name="FC"),
        rate_constant1.make_node(name="rate_constant1"),
        rate_constant2.make_node(name="rate_constant2"),
        param_head.make_node(name="param_head"),
    ] + Multi_Reaction.make_nodes()

    # -------- TRAINING --------
    domain = Domain()
    
    # Common input dictionary for all constraints
    input_dict = {
        "x": x_aug, 
        "cid": cid_aug,
        "B_init_in": B_init_aug,
        "DT_in": DT_aug,
        "c1": ones, "c2": ones, "cA": ones, # Dummy inputs for parameters
    }

    # 1. PDE Residuals
    domain.add_constraint(
        PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar=input_dict,
            outvar={"reaction_R_norm": zeros, "reaction_R2_norm": zeros, "order_norm": zeros},
            batch_size=getattr(cfg.batch_size, "residual_R", 1024),
        ),
        "residual",
    )

    # 2. Data Fit
    # 需要加入 B_meas
    data_fit_input = input_dict.copy()
    data_fit_input["B_meas"] = B_meas_aug
    
    domain.add_constraint(
        PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar=data_fit_input,
            outvar={"stoich_norm": zeros},
            batch_size=getattr(cfg.batch_size, "stoich", 1024),
        ),
        "stoich",
    )

    total_steps = int(getattr(cfg.training, "max_steps", 200000))
    p(f"[TRAINING] Multi-experiment training for {total_steps} steps...")
    solver = Solver(cfg, domain)
    solver.solve()
    p("[DONE] Training")

    # -------- POST-PROCESSING --------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def sigmoid_np(z): return 1.0/(1.0+np.exp(-z))

        # 1. Extract Shared Parameters
        m1 = get_core_module(find_torch_module(rate_constant1)); m1.eval()
        m2 = get_core_module(find_torch_module(rate_constant2)); m2.eval()
        mph = get_core_module(find_torch_module(param_head)); mph.eval()
        
        # dummy input
        dummy_in = torch.tensor([[1.0]], dtype=torch.float32, device=module_device(m1))
        
        with torch.no_grad():
            raw12 = m1(dummy_in).cpu().numpy().ravel()
            raw34 = m2(dummy_in).cpu().numpy().ravel()
            rawAh = mph(dummy_in).cpu().numpy().ravel()

        k1 = float(K1_LO) + (float(K1_HI) - float(K1_LO))*sigmoid_np(raw12[0])
        k2 = float(K2_LO) + (float(K2_HI) - float(K2_LO))*sigmoid_np(raw12[1])
        k3 = float(K3_LO) + (float(K3_HI) - float(K3_LO))*sigmoid_np(raw34[0])
        k4 = float(K4_LO) + (float(K4_HI) - float(K4_LO))*sigmoid_np(raw34[1])

        A0 = float(A0_LO) + (float(A0_HI) - float(A0_LO))*sigmoid_np(rawAh[0])
        C0 = float(C0_LO) + (float(C0_HI) - float(C0_LO))*sigmoid_np(rawAh[1])
        D0 = float(D0_LO) + (float(D0_HI) - float(D0_LO))*sigmoid_np(rawAh[2])
        E0 = float(E0_LO) + (float(E0_HI) - float(E0_LO))*sigmoid_np(rawAh[3])

        p(f"[POST] SHARED PARAMS:")
        p(f"  k1={k1:.5e}, k2={k2:.5e}, k3={k3:.5e}, k4={k4:.5e}")
        p(f"  A0={A0:.4f}, C0={C0:.4f}, D0={D0:.4f}, E0={E0:.4f}")

        # 2. ODE Solver for Verification
        def solve_ode_rk4(t_arr, B_init_val):
            # t_arr is real time (seconds) relative to start
            dt_steps = t_arr[1:] - t_arr[:-1]
            R_sim  = np.zeros_like(t_arr)
            R2_sim = np.zeros_like(t_arr)
            # IC is 0, 0
            
            delta = 1e-10
            
            for i in range(len(dt_steps)):
                dt = float(dt_steps[i])
                
                def rhs(r, r2):
                    H_raw = B_init_val - r - 9.0*r2
                    H_pos = (H_raw + np.sqrt(H_raw**2 + delta))/2.0 + EPS
                    
                    dR  = k1*(A0 - r)*H_pos - k2*(C0 + r + 9.0*r2)
                    dR2 = k3*(D0 - r2)*(H_pos**9.0) - k4*((C0 + r + 9.0*r2)**9.0)*((E0 + 4.0*r2)**4.0)
                    return dR, dR2
                
                cur_r, cur_r2 = R_sim[i], R2_sim[i]
                
                k1r, k1r2 = rhs(cur_r, cur_r2)
                k2r, k2r2 = rhs(cur_r + 0.5*dt*k1r, cur_r2 + 0.5*dt*k1r2)
                k3r, k3r2 = rhs(cur_r + 0.5*dt*k2r, cur_r2 + 0.5*dt*k2r2)
                k4r, k4r2 = rhs(cur_r + dt*k3r,     cur_r2 + dt*k3r2)
                
                R_sim[i+1]  = np.clip(cur_r  + (dt/6.0)*(k1r + 2*k2r + 2*k3r + k4r), 0, B_init_val)
                R2_sim[i+1] = np.clip(cur_r2 + (dt/6.0)*(k1r2 + 2*k2r2 + 2*k3r2 + k4r2), 0, D0)
            
            H_sim = np.maximum(B_init_val - R_sim - 9.0*R2_sim, 0.0)
            return H_sim, R_sim, R2_sim

        # 3. Validation Plots (Per Experiment)
        artifacts = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts, exist_ok=True)
        
        mfc = get_core_module(find_torch_module(FC)); mfc.eval()

        for meta in exp_meta:
            cid = meta["cid"]
            p(f"[CHECK] Processing Exp {cid} ...")
            
            # A. NN Prediction
            # Construct input tensor: [x, cid]
            nx = meta["x"]
            xin = np.stack([nx, np.full_like(nx, cid)], axis=1) # shape (N, 2)
            
            with torch.no_grad():
                out = mfc(torch.tensor(xin, dtype=torch.float32, device=module_device(mfc))).cpu().numpy()
            
            # Decode NN output (using HARD IC logic)
            B_init_v = meta["B_init"]
            R_nn  = nx * (1.0/(1.0+np.exp(-np.clip(out[:,0],-60,60)))) * B_init_v
            R2_nn = nx * (1.0/(1.0+np.exp(-np.clip(out[:,1],-60,60)))) * D0
            H_nn  = np.maximum(B_init_v - R_nn - 9.0*R2_nn, 0.0)

            # B. ODE Simulation (Verification)
            t_rel = meta["t_raw"] - meta["t0"]
            H_ode, R_ode, R2_ode = solve_ode_rk4(t_rel, B_init_v)
            
            # C. Plot
            fig, ax = plt.subplots(dpi=150)
            ax.plot(meta["t_raw"], meta["B_meas"], "ko", ms=2, alpha=0.5, label="Data")
            ax.plot(meta["t_raw"], H_nn, "b--", lw=2, label="NN Prediction")
            ax.plot(meta["t_raw"], H_ode, "r-", lw=1.5, label="ODE (Shared k)")
            
            ax.set_title(f"Experiment {int(cid)} Fit (Shared Parameters)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("H Concentration")
            ax.legend()
            ax.grid(True)
            
            fname = os.path.join(artifacts, f"Exp_{int(cid)}_fit.png")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            p(f"  -> Saved {fname}")
            
            # Save CSV
            out_df = pd.DataFrame({
                "time": meta["t_raw"],
                "H_data": meta["B_meas"],
                "H_nn": H_nn,
                "H_ode": H_ode,
                "R_ode": R_ode,
                "R2_ode": R2_ode
            })
            out_df.to_csv(os.path.join(artifacts, f"Exp_{int(cid)}_pred.csv"), index=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        p("[POST] Failed:", e)

if __name__ == "__main__":
    run()
