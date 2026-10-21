"""Tests for the PCFM release package. Runs with a randomly initialized
checkpoint saved in the published format (no network needed) plus a real
statistical test of the "CP" graph estimator on known synthetic physics.
Run:  python test_release.py"""
import os
import tempfile

import numpy as np
import pandas as pd
import torch

from model import CONFIG, PCFM, PCFMMAEWrapper, PCFMPredictor

torch.manual_seed(0)
np.random.seed(0)

CKPT = os.path.join(tempfile.gettempdir(), "pcfm_test_ckpt.pt")

# --- 1. fabricate a checkpoint in the published format and load it strictly ---
backbone = PCFM(dim=CONFIG["dim"], patch_size=CONFIG["patch_size"], layers=CONFIG["layers"],
                num_heads=CONFIG["num_heads"], max_context_steps=CONFIG["max_context_steps"],
                spatial_stride=CONFIG["spatial_stride"])
w = PCFMMAEWrapper(backbone, CONFIG["dim"], CONFIG["patch_size"], CONFIG["pred_len"])
torch.save({"config": {"dim": CONFIG["dim"], "patch_size": CONFIG["patch_size"],
                       "num_layers": CONFIG["layers"], "num_heads": CONFIG["num_heads"],
                       "max_context_steps": CONFIG["max_context_steps"],
                       "pred_len": CONFIG["pred_len"]},
            "model_state_dict": w.state_dict()}, CKPT)
model = PCFMPredictor.from_pretrained(CKPT, device="cpu")
print("PASS  strict checkpoint load (.pt)")

# --- 1b. safetensors format (the format published on the Hugging Face Hub) ---
from safetensors.torch import save_file
ST = os.path.join(tempfile.gettempdir(), "pcfm_test_ckpt.safetensors")
save_file(w.state_dict(), ST)
model_st = PCFMPredictor.from_pretrained(ST, device="cpu")
_h = np.random.randn(2, 256).astype(np.float32)
assert torch.allclose(model.forecast(_h, 32), model_st.forecast(_h, 32)), \
    "safetensors and .pt checkpoints must give identical outputs"
os.remove(ST)
print("PASS  safetensors checkpoint load (identical outputs)")

N, CONTEXT, HORIZON = 2, 512, 128
P = model.patch_size
hist = np.random.randn(N, CONTEXT).astype(np.float32).cumsum(axis=1)

# --- 2. forecast shapes + quantile monotonicity + graph modes ---
med = model.forecast(hist, HORIZON)                                  # default = "CD"
assert med.shape == (N, HORIZON)
q = model.forecast(hist, HORIZON, return_quantiles=True)
assert q.shape == (N, HORIZON, 9)
assert (q[..., 1:] - q[..., :-1] >= -1e-5).all(), "quantiles must be monotone"
med_ci = model.forecast(hist, HORIZON, graph="CI")
med_cd = model.forecast(hist, HORIZON, graph="CD")
assert torch.allclose(med_cd, med), '"CD" must equal the default'
assert (med_ci - med_cd).abs().max() > 1e-6, "CI and CD must differ (graph plumbing)"
print("PASS  forecast shapes + monotone quantiles + CI/CD modes")

# --- 3. determinism ---
assert torch.allclose(med, model.forecast(hist, HORIZON)), "same input, same output"
print("PASS  determinism")

# --- 4. custom graphs: tuples and [2, E]; lags must be plumbed ---
med_t = model.forecast(hist, HORIZON, graph=[(0, 1, 0.0)])           # tuple form (+auto self-loops)
med_a = model.forecast(hist, HORIZON, graph=[[0, 0, 1], [0, 1, 1]], edge_lags=[0, 0, 0])
assert torch.allclose(med_t, med_a, atol=1e-5), "tuple graph must equal explicit [2,E] graph"
med_lag = model.forecast(hist, HORIZON, graph=[(0, 1, 64.0)])
d_lag = (med_t[1] - med_lag[1]).abs().max().item()
assert d_lag > 1e-6, "edge lag had no effect -> lags not plumbed"
assert torch.allclose(med_t[0], med_lag[0]), "lag on 0->1 must not affect node 0"
print(f"PASS  custom graphs (tuple == [2,E]; lag effect {d_lag:.4f})")

# --- 5. CP mode: recover the causal edge AND its lag from known physics ---
LAG = 64
rng = np.random.RandomState(3)
flips = rng.rand(4096) < 0.02
state, sig = 0, []
for f in flips:
    if f: state = 1 - state
    sig.append(state)
valve = np.array(sig, dtype=float)
temp, y = np.empty(4096), 20.0
for t in range(4096):
    u = valve[t - LAG] if t >= LAG else 0.0
    y = 0.95 * y + 0.05 * (20.0 + 60.0 * u)
    temp[t] = y
temp += rng.normal(0, 0.3, 4096)
sysX = np.stack([valve, temp])

ei, ea = model.estimate_graph(sysX)
pairs = {(int(s), int(d)): float(l) for s, d, l in zip(ei[0], ei[1], ea)}
assert (0, 0) in pairs and (1, 1) in pairs, "self-loops missing"
assert (0, 1) in pairs, "true edge valve->temp not recovered"
assert abs(pairs[(0, 1)] - LAG) <= 32, f"lag estimate {pairs[(0, 1)]} too far from true {LAG}"
med_cp = model.forecast(sysX[:, :1024], HORIZON, graph="CP")
assert med_cp.shape == (2, HORIZON)
print(f"PASS  CP estimation (edges {sorted(pairs)}; lag(0->1) = {pairs[(0, 1)]:.0f} ~ {LAG})")

# --- 6. what_if: mask semantics, glue, plan plumbing ---
window = np.random.randn(N, CONTEXT + HORIZON).astype(np.float32).cumsum(axis=1)
recon = model.what_if(window, context_length=CONTEXT, masked_nodes=[1], graph=[(0, 1, 16.0)])
wt = torch.as_tensor(window)
assert recon.shape == (N, CONTEXT + HORIZON)
assert torch.allclose(recon[0], wt[0], atol=1e-5), "unmasked node must stay observed"
assert torch.allclose(recon[1, :CONTEXT], wt[1, :CONTEXT], atol=1e-5), "context must stay observed"
assert (recon[1, CONTEXT:] - wt[1, CONTEXT:]).abs().max() > 1e-4, "masked future must be predicted"
assert abs(recon[1, CONTEXT].item() - window[1, CONTEXT - 1]) < 1e-4, "glue must anchor the forecast"

w_on, w_off = window.copy(), window.copy()
w_on[0, CONTEXT:] = 5.0
w_off[0, CONTEXT:] = -5.0
r_on = model.what_if(w_on, context_length=CONTEXT, masked_nodes=[1], graph=[(0, 1, 16.0)])
r_off = model.what_if(w_off, context_length=CONTEXT, masked_nodes=[1], graph=[(0, 1, 16.0)])
d_plan = (r_on[1, CONTEXT:] - r_off[1, CONTEXT:]).abs().max().item()
assert d_plan > 1e-4, "changing the driver plan must change the masked prediction"
print(f"PASS  what_if mask + glue + plan plumbing (effect {d_plan:.4f})")

# --- 7. arbitrary lengths (auto patch-padding) ---
w_odd = np.random.randn(N, 1000 + 24).astype(np.float32).cumsum(axis=1)
r_odd = model.what_if(w_odd, context_length=1000, masked_nodes=[1])
assert r_odd.shape == (N, 1024)
assert torch.allclose(r_odd[1, :1000], torch.as_tensor(w_odd)[1, :1000], atol=1e-5)
assert (r_odd[1, 1000:] - torch.as_tensor(w_odd)[1, 1000:]).abs().max() > 1e-4
print("PASS  auto patch-padding for arbitrary context/horizon lengths")

# --- 8. custom mask (patch- and step-level) ---
P_total = (CONTEXT + HORIZON) // P
m_patch = np.zeros((N, P_total)); m_patch[1, -2:] = 1.0
r1 = model.what_if(window, custom_mask=m_patch, graph=[(0, 1, 16.0)])
assert torch.allclose(r1[1, :-(2 * P)], wt[1, :-(2 * P)], atol=1e-5)
assert (r1[1, -(2 * P):] - wt[1, -(2 * P):]).abs().max() > 1e-4
m_step = np.zeros((N, CONTEXT + HORIZON)); m_step[1, -2 * P:] = 1.0
r2 = model.what_if(window, custom_mask=m_step, graph=[(0, 1, 16.0)])
assert torch.allclose(r1, r2), "step-level mask must equal its patch-level pooling"
print("PASS  custom mask (patch- and step-level)")

# --- 9. DataFrame API ---
ts = pd.date_range("2024-01-01", periods=1000, freq="h")
frames = []
for gid in ("plant_A", "plant_B"):
    off = 0 if gid == "plant_A" else 500
    frames.append(pd.DataFrame({"id": gid, "timestamp": ts,
                                "temp": sysX[1, off:off + 1000],
                                "valve": sysX[0, off:off + 1000]}))
ctx_df = pd.concat(frames, ignore_index=True)

# 9a. plain forecast (no future_df), quantile interpolation
pred = model.predict_df(ctx_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9],
                        target="temp", graph=[("valve", "temp", 64.0)])
assert len(pred) == 2 * 24
assert set(["id", "timestamp", "target_column", "predictions", "0.1", "0.5", "0.9"]) <= set(pred.columns)
assert (pred["0.1"] <= pred["0.5"] + 1e-5).all() and (pred["0.5"] <= pred["0.9"] + 1e-5).all()
assert np.allclose(pred["0.5"], pred["predictions"], atol=1e-5), "0.5 must equal the median"
a_rows = pred[pred["id"] == "plant_A"]
assert list(a_rows["timestamp"])[0] == ts[-1] + pd.Timedelta(hours=1), "future timestamps must continue the context"
print("PASS  predict_df forecast (2 ids, quantile interpolation, timestamps)")

# 9b. future_df -> what-if: valve plan must steer the temp prediction
fut_ts = pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
fut_on = pd.concat([pd.DataFrame({"id": g, "timestamp": fut_ts, "valve": 1.0}) for g in ("plant_A", "plant_B")])
fut_off = pd.concat([pd.DataFrame({"id": g, "timestamp": fut_ts, "valve": 0.0}) for g in ("plant_A", "plant_B")])
p_on = model.predict_df(ctx_df, future_df=fut_on, prediction_length=24, target="temp",
                        graph=[("valve", "temp", 64.0)])
p_off = model.predict_df(ctx_df, future_df=fut_off, prediction_length=24, target="temp",
                         graph=[("valve", "temp", 64.0)])
d_df = np.abs(np.asarray(p_on["predictions"]) - np.asarray(p_off["predictions"])).max()
assert d_df > 1e-4, "future_df covariate plan must reach the prediction"
assert list(p_on["timestamp"][:24]) == list(fut_ts), "future_df timestamps must be used"
print(f"PASS  predict_df what-if via future_df (plan effect {d_df:.4f})")

# 9c. multiple targets + no id column
single = ctx_df[ctx_df["id"] == "plant_A"].drop(columns="id")
p_multi = model.predict_df(single, prediction_length=24, target=["temp", "valve"], graph="CD")
assert len(p_multi) == 2 * 24 and set(p_multi["target_column"]) == {"temp", "valve"}
print("PASS  predict_df multiple targets / single unnamed series")

# --- 10. input validation ---
for bad_call in (lambda: model.what_if(window, context_length=CONTEXT + HORIZON + 5, masked_nodes=[1]),
                 lambda: model.forecast(np.full((2, 64), np.nan)),
                 lambda: model.forecast(hist, 32, graph=[[0], [5]]),
                 lambda: model.forecast(hist, 32, graph="XX"),
                 lambda: model.predict_df(ctx_df, prediction_length=24, target="nope"),
                 lambda: model.predict_df(ctx_df, future_df=fut_on.iloc[:5], prediction_length=24, target="temp")):
    try:
        bad_call()
        raise SystemExit(f"expected a ValueError from {bad_call}")
    except ValueError:
        pass
print("PASS  input validation")

# --- 11. batch padding (m_pad_step): a left-padded series in a batch == the same series alone ---
from model import GraphInput
_a = np.random.randn(1, 320).astype(np.float32).cumsum(-1)
_b = np.random.randn(1, 512).astype(np.float32).cumsum(-1)
with torch.no_grad():
    alone = model.model(GraphInput(x=torch.from_numpy(_a), edge_index=torch.zeros(2, 1, dtype=torch.long),
                                   edge_attr=torch.zeros(1, 1), batch=torch.zeros(1, dtype=torch.long), num_graphs=1),
                        mode="forecast", pred_len=64)
    X = np.zeros((2, 512), np.float32); X[0, 192:] = _a[0]; X[1] = _b[0]
    pad = np.zeros((2, 512, 1), np.float32); pad[0, :192] = 1.0
    both = model.model(GraphInput(x=torch.from_numpy(X), edge_index=torch.arange(2).repeat(2, 1),
                                  edge_attr=torch.zeros(2, 1), batch=torch.arange(2), num_graphs=2,
                                  m_pad_step=torch.from_numpy(pad)), mode="forecast", pred_len=64)
assert torch.allclose(alone[0], both[0], rtol=1e-4, atol=1e-4), "batch padding must be invisible to the forecast"
print("PASS  batch padding mask (m_pad_step)")

os.remove(CKPT)
print("\nALL RELEASE TESTS PASSED")
