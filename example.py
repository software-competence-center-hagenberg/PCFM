"""PCFM end-to-end demo: forecasting, what-if interventions, graph modes,
custom masks, and the pandas API -- each demo prints its numbers AND saves a
figure, all on synthetic systems with known physics so every output can be
checked against the analytic ground truth.

The demos replicate the counterfactual protocol of the accompanying
dissertation's what-if suite (context 4096, horizon 128, bipolar drivers,
noise 0.5, do-operator masking, intervention-effect metric, and the suite's
visual grammar: ground-truth halo bands, dashed t=0 line, shaded prediction
horizon, black observed history, plans drawn from t=0 only).

    demo 1  what-if      : Valve ON / OFF / PULSE plans, temperature masked,
                           checked against analytic physics
                           -> example_whatif.png
    demo 2  graph modes  : a SPARSE 10-variable system (four independent
                           units, six causal edges -- the dissertation's
                           topology-ablation benchmark): true graph vs
                           CP (estimated) vs CD vs CI
                           -> example_graph_modes.png
    demo 3  custom mask  : in-fill a past gap AND predict the future with one
                           hand-built mask         -> example_custom_mask.png
    demo 4  pandas API   : predict_df with future_df carrying the valve plan,
                           quantile fan chart      -> example_predict_df.png
    demo 5  forecast     : (a) median + 10-90% band when the driver's FUTURE
                           is unknown (contrast with demo 1, where a plan is
                           given)                  -> example_forecast.png
                           (b) two COUPLED CONTINUOUS variates predicted
                           jointly                 -> example_forecast_coupled.png

Run:  python example.py [checkpoint_path_or_repo]
      (defaults to the Hugging Face repo software-competence-center-hagenberg/PCFM)
"""
import sys

import numpy as np
import torch

from model import PCFMPredictor

SEED = 7
LAG = 48                       # valve->temp transport delay: 1.5 patches
CONTEXT, HORIZON = 4096, 128   # = the dissertation suite's seq_len / pred_len
NOISE = 0.5
PLOT_CONTEXT = 512             # display crop, exactly like the suite's figures
rng = np.random.RandomState(SEED)

mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# =========================================================================
# Synthetic physics (known in closed form)
# =========================================================================
def telegraph(rand, n, p=0.02):
    """Binary random telegraph signal in {0, 1}."""
    flips = rand.rand(n) < p
    state, sig = 0, np.empty(n)
    for i, f in enumerate(flips):
        if f:
            state = 1 - state
        sig[i] = state
    return sig


def fo_response(parent, lag, alpha, offset, slope):
    """First-order lag toward setpoint = offset + slope * parent(t - lag)."""
    p_del = np.roll(parent, lag)
    p_del[:lag] = parent[0]
    out, val = np.empty(len(parent)), offset + slope * p_del[0]
    for t in range(len(parent)):
        val = alpha * val + (1 - alpha) * (offset + slope * p_del[t])
        out[t] = val
    return out


def simulate_valve_temp(valve_bipolar, temp0=20.0):
    """Clean valve->temp physics: setpoint 20 + 60*(valve+1)/2, delay LAG."""
    temp, y = np.empty(len(valve_bipolar)), temp0
    for t in range(len(valve_bipolar)):
        u = valve_bipolar[t - LAG] if t >= LAG else -1.0
        y = 0.95 * y + 0.05 * (20.0 + 60.0 * (u + 1.0) / 2.0)
        temp[t] = y
    return temp


def make_valve_temp_window():
    """[2, CONTEXT+HORIZON] window, chosen like the suite's base windows:
    valve OFF over the last 64 steps (clean step onset), well-excited before."""
    total = 20000
    v01 = telegraph(rng, total)
    valve = 2.0 * v01 - 1.0
    temp = simulate_valve_temp(valve) + rng.normal(0, NOISE, total)
    T = CONTEXT + HORIZON
    for i in range(total - T):
        w01 = v01[i:i + T]
        if w01[CONTEXT - 64:CONTEXT].mean() < 0.1 and 0.2 < w01[:CONTEXT].mean() < 0.8:
            return np.stack([valve[i:i + T], temp[i:i + T]])
    return np.stack([valve[:T], temp[:T]])


def make_coupled_continuous_window():
    """Two coupled CONTINUOUS variates, both forecastable and predicted JOINTLY:
    node 0  drive -- smooth quasi-periodic signal (two incommensurate periods)
    node 1  resp  -- first-order response to the drive through the same 48-step
                     transport delay (setpoint 30 + 20 * drive)"""
    r = np.random.RandomState(SEED + 2)
    t = np.arange(CONTEXT + HORIZON)
    drive = (0.9 * np.sin(2 * np.pi * t / 256) + 0.5 * np.sin(2 * np.pi * t / 384 + 1.0)
             + r.normal(0, 0.05, len(t)))
    resp = fo_response(drive, LAG, 0.95, 30.0, 20.0) + r.normal(0, NOISE, len(t))
    return np.stack([drive, resp])


# The dissertation's 10-variable topology-ablation system: four independent
# units at heterogeneous scales, six causal edges. Node 0 is the intervention
# driver, node 1 the target (identical physics to the valve benchmark).
SPARSE_MECHANISMS = [   # (node, parent, lag, alpha, offset, slope)
    (1, 0, 16, 0.95, 20.0, 60.0),    # u0 -> y1   THE INTERVENED MECHANISM
    (3, 2, 24, 0.95, 100.0, 100.0),  # u2 -> y3
    (5, 4, 8, 0.90, 0.0, 10.0),      # u4 -> y5
    (6, 1, 16, 0.90, 0.0, 0.05),     # y1 -> y6   (downstream of the target)
    (7, 3, 32, 0.92, 0.0, 2.0),      # y3 -> y7
    (9, 8, 12, 0.95, -50.0, 30.0),   # u8 -> y9
]
SPARSE_NOISE = {1: 0.5, 3: 1.0, 5: 0.1, 6: 0.03, 7: 3.0, 9: 0.4}
SPARSE_DRIVERS, SPARSE_RESPONSES = [0, 2, 4, 8], [1, 3, 5, 6, 7, 9]


def make_sparse_window():
    """[10, CONTEXT+HORIZON] window of the sparse system + its true graph.
    Node 0 (the intervention driver) is encoded bipolar, as in the suite."""
    r = np.random.RandomState(SEED + 1)
    total = CONTEXT + HORIZON
    data = np.zeros((total, 10))
    for d in SPARSE_DRIVERS:
        data[:, d] = telegraph(r, total)
    clean = {}
    for node, parent, lg, alpha, off, slope in SPARSE_MECHANISMS:
        parent_sig = clean.get(parent, data[:, parent])
        clean[node] = fo_response(parent_sig, lg, alpha, off, slope)
        data[:, node] = clean[node] + r.normal(0, SPARSE_NOISE[node], total)
    window = data.T.copy()
    window[0] = 2.0 * window[0] - 1.0                     # bipolar intervention driver
    true_graph = [(p, n, float(lg)) for n, p, lg, _, _, _ in SPARSE_MECHANISMS]
    return window, true_graph


def simulate_sparse_target(u0_bipolar, y0):
    """Clean-physics roll-out of the sparse system's target y1 for a given
    bipolar u0 trajectory (as in the suite's topology experiment)."""
    u = (np.asarray(u0_bipolar, dtype=float) + 1.0) / 2.0
    _, _, lg, alpha, off, slope = SPARSE_MECHANISMS[0]
    out, y = np.empty(len(u)), float(y0)
    for t in range(len(u)):
        y = alpha * y + (1 - alpha) * (off + slope * u[max(t - lg, 0)])
        out[t] = y
    return out


# =========================================================================
# Plot helpers (the suite's visual grammar)
# =========================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'grid.alpha': 0.3, 'grid.linestyle': ':', 'axes.axisbelow': True})
    return plt


HALO = dict(lw=6.0, alpha=0.45, solid_capstyle='round', zorder=1.8)


def _mark(ax, horizon=HORIZON, annotate=False, H=PLOT_CONTEXT):
    import matplotlib.transforms as mtransforms
    ax.axvspan(0, horizon, color='0.45', alpha=0.08, zorder=0)
    ax.axvline(0, color='0.15', lw=1.3, ls=(0, (4, 2.5)), zorder=6)
    ax.grid(True)
    if annotate:
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(-H / 2, 1.02, "Context (observed)", transform=tr, ha='center',
                va='bottom', fontsize=9, color='0.3', fontstyle='italic')
        ax.text(horizon / 2, 1.02, "Prediction horizon (t ≥ 0)", transform=tr,
                ha='center', va='bottom', fontsize=9, color='0.3', fontstyle='italic')


def _tag(ax, role):
    ax.text(1.008, 0.5, role, transform=ax.transAxes, rotation=270, ha='left',
            va='center', fontsize=11, color='0.15', fontstyle='italic')


def _zscale(hist_vals):
    mu, sd = np.mean(hist_vals), np.std(hist_vals) + 1e-9
    return lambda x: (np.asarray(x) - mu) / sd


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=200)
    print(f"Plot saved: {name}")


# =========================================================================
# Demos
# =========================================================================
def demo5_forecast(model, window, graph):
    print("\n[demo 5] forecast: 128 steps ahead, future valve UNKNOWN")
    history, truth = window[:, :CONTEXT], window[:, CONTEXT:]
    q = model.forecast(history, HORIZON, graph=graph, return_quantiles=True)
    med = q[1, :, model.median_idx].cpu().numpy()
    lo, hi = q[1, :, 0].cpu().numpy(), q[1, :, -1].cpu().numpy()
    cover = float(np.mean((truth[1] >= lo) & (truth[1] <= hi)))
    print(f"  temperature MAE vs held-out truth: {mae(med, truth[1]):.3f} degC")
    print(f"  truth inside the 10-90% band     : {cover * 100:.0f}%")
    ei, ea = model.estimate_graph(history)
    print(f"  estimate_graph recovered: "
          f"{[(int(s), int(d), float(l)) for s, d, l in zip(ei[0], ei[1], ea) if s != d]} "
          f"(true: 0->1 @ {LAG})")

    plt = _plt()
    t_h, t_f = np.arange(-PLOT_CONTEXT, 0), np.arange(HORIZON)
    z = _zscale(window[1, :CONTEXT])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 5.8), sharex=True,
                                   height_ratios=[1.6, 1])
    ax1.plot(t_h, z(window[1, CONTEXT - PLOT_CONTEXT:CONTEXT]), 'k-', lw=2.2, label="History")
    ax1.plot(t_f, z(truth[1]), color='tab:red', label="Ground truth (held out)", **HALO)
    ax1.fill_between(t_f, z(lo), z(hi), color='tab:red', alpha=0.18,
                     label="Forecast 10–90% band")
    ax1.plot(t_f, z(med), color='tab:red', lw=2.2, zorder=3, label="Forecast (median)")
    _mark(ax1, annotate=True)
    _tag(ax1, "prediction")
    ax1.set_ylabel("Temp (z)")
    ax1.set_title("Forecast · the valve's future is UNKNOWN here", pad=20)
    ax1.legend(loc='upper left', ncol=2, fontsize=8)
    ax2.step(t_h, window[0, CONTEXT - PLOT_CONTEXT:CONTEXT], where='post',
             color='black', lw=2.2, label="History")
    ax2.step(t_f, truth[0], where='post', color='0.55', lw=2.0, ls='--',
             label="Actual future (not shown to the model)")
    _mark(ax2)
    _tag(ax2, "unknown future")
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_ylabel("Valve [-1, 1]")
    ax2.set_xlabel("Time step")
    ax2.legend(loc='upper left', ncol=2, fontsize=8)
    _save(fig, "example_forecast.png")
    plt.close(fig)

    # ---- part B: two COUPLED CONTINUOUS variates, both predicted jointly ----
    print("\n[demo 5b] forecast: two coupled continuous variates, BOTH predicted")
    cw = make_coupled_continuous_window()
    c_hist, c_truth = cw[:, :CONTEXT], cw[:, CONTEXT:]
    qc = model.forecast(c_hist, HORIZON, graph=[(0, 1, float(LAG))],
                        return_quantiles=True)
    labels = ("Drive", "Response")
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 5.8), sharex=True)
    t_h = np.arange(-PLOT_CONTEXT, 0)
    t_f = np.arange(HORIZON)
    for i, (ax, lbl, color) in enumerate(zip(axes, labels, ('tab:orange', 'tab:red'))):
        med = qc[i, :, model.median_idx].cpu().numpy()
        lo, hi = qc[i, :, 0].cpu().numpy(), qc[i, :, -1].cpu().numpy()
        cover = float(np.mean((c_truth[i] >= lo) & (c_truth[i] <= hi)))
        print(f"  {lbl:9s}: MAE {mae(med, c_truth[i]):6.3f} | truth inside 10-90% band {cover*100:3.0f}%")
        z = _zscale(cw[i, :CONTEXT])
        ax.plot(t_h, z(cw[i, CONTEXT - PLOT_CONTEXT:CONTEXT]), 'k-', lw=2.0, label="History")
        ax.plot(t_f, z(c_truth[i]), color=color, label="Ground truth (held out)", **HALO)
        ax.fill_between(t_f, z(lo), z(hi), color=color, alpha=0.18, label="10–90% band")
        ax.plot(t_f, z(med), color=color, lw=2.2, zorder=3, label="Forecast (median)")
        _mark(ax, annotate=(i == 0))
        _tag(ax, "prediction")
        ax.set_ylabel(f"{lbl} (z)")
        ax.legend(loc='upper left', ncol=4, fontsize=8)
    axes[0].set_title("Joint forecast · two coupled continuous variates "
                      "(drive → response @ lag 48)", pad=20)
    axes[1].set_xlabel("Time step")
    _save(fig, "example_forecast_coupled.png")
    plt.close(fig)


def demo1_whatif(model, window, graph):
    print("\n[demo 1] what-if: do(valve := plan) for t >= 0, temperature masked")
    on, off = np.ones(HORIZON), -np.ones(HORIZON)
    pulse = np.concatenate([np.ones(32), -np.ones(HORIZON - 32)])
    scenarios = {"Valve ON": (on, 'tab:red', '-'),
                 "Valve OFF": (off, 'tab:blue', '--'),
                 "Valve PULSE": (pulse, 'tab:green', ':')}
    results = {}
    for name, (plan, color, ls) in scenarios.items():
        w = window.copy()
        w[0, CONTEXT:] = plan                                # the intervention
        truth = simulate_valve_temp(w[0], temp0=w[1, 0])[CONTEXT:]
        recon = model.what_if(w, context_length=CONTEXT, masked_nodes=[1], graph=graph)
        pred = recon[1, CONTEXT:].cpu().numpy()
        results[name] = (plan, pred, truth, color, ls)
        print(f"  {name:12s}: predicted end {pred[-1]:6.2f} degC | physics end "
              f"{truth[-1]:6.2f} degC | MAE {mae(pred, truth):6.3f}")
    eff_p = results["Valve ON"][1] - results["Valve OFF"][1]
    eff_t = results["Valve ON"][2] - results["Valve OFF"][2]
    print(f"  intervention-effect error (MAE of ON-minus-OFF): {mae(eff_p, eff_t):6.3f}")

    plt = _plt()
    t_h, t_f = np.arange(-PLOT_CONTEXT, 0), np.arange(HORIZON)
    z = _zscale(window[1, :CONTEXT])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 5.8), sharex=True)
    ax1.plot(t_h, z(window[1, CONTEXT - PLOT_CONTEXT:CONTEXT]), 'k-', lw=2.2, label="History")
    ax2.step(t_h, window[0, CONTEXT - PLOT_CONTEXT:CONTEXT], where='post',
             color='black', lw=2.2, label="History")
    for name, (plan, pred, truth, color, ls) in results.items():
        ax1.plot(t_f, z(truth), color=color, label=f"Ground truth ({name})", **HALO)
        ax1.plot(t_f, z(pred), color=color, ls=ls, lw=2.4, zorder=3,
                 label=f"Prediction ({name})")
        ax2.step(np.arange(-1, HORIZON), np.concatenate([[window[0, CONTEXT - 1]], plan]),
                 where='post', color=color, ls=ls, lw=2.4, label=f"Plan: {name}")
    _mark(ax1, annotate=True)
    _mark(ax2)
    _tag(ax1, "prediction")
    _tag(ax2, "intervention")
    ax1.set_ylabel("Temp (z)")
    ax1.set_title("Zero-shot what-if · valve → temperature", pad=20)
    ax1.legend(loc='upper left', ncol=3, fontsize=8)
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_ylabel("Valve [-1, 1]")
    ax2.set_xlabel("Time step")
    ax2.legend(loc='upper left', ncol=4, fontsize=8)
    _save(fig, "example_whatif.png")
    plt.close(fig)
    return results


def demo2_graph_modes(model, ckpt_note=""):
    print("\n[demo 2] graph modes on the SPARSE 10-variable system "
          "(4 independent units, 6 causal edges)")
    window, true_graph = make_sparse_window()
    on, off = np.ones(HORIZON), -np.ones(HORIZON)
    conditions = {"True graph": true_graph, "CP (estimated)": "CP",
                  "CD (fully connected)": "CD", "CI (independent)": "CI"}
    # leakage guard from the suite: mask ALL response nodes (y6 is downstream
    # of the target y1 -- its observed future would leak the answer back)
    masked = SPARSE_RESPONSES

    truths, preds = {}, {c: {} for c in conditions}
    for level, name in ((1.0, "ON"), (-1.0, "OFF")):
        w = window.copy()
        w[0, CONTEXT:] = level
        truths[name] = simulate_sparse_target(w[0], y0=w[1, 0])[CONTEXT:]
        for cond, g in conditions.items():
            r = model.what_if(w, context_length=CONTEXT, masked_nodes=masked, graph=g)
            preds[cond][name] = r[1, CONTEXT:].cpu().numpy()

    ci_limit = mae(truths["ON"], truths["OFF"])   # best a graph-blind model can do
    print(f"  theoretical CI limit (mean|truth_ON - truth_OFF|): {ci_limit:.3f} degC")
    effect_err = {}
    for cond in conditions:
        effect_err[cond] = mae(preds[cond]["ON"] - preds[cond]["OFF"],
                               truths["ON"] - truths["OFF"])
        print(f"  intervention-effect error  {cond:22s}: {effect_err[cond]:7.3f} degC")

    plt = _plt()
    z = _zscale(window[1, :CONTEXT])
    t_f = np.arange(HORIZON)
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for j, (ax, cond) in enumerate(zip(axes, conditions)):
        if j > 0:
            ax.sharey(axes[0])
        for name, color, ls in (("ON", 'tab:red', '-'), ("OFF", 'tab:blue', '--')):
            ax.plot(t_f, z(truths[name]), color=color,
                    label=f"Ground truth (Valve {name})", **HALO)
            ax.plot(t_f, z(preds[cond][name]), color=color, ls=ls, lw=2.4, zorder=3,
                    label=f"Prediction (Valve {name})")
        _mark(ax)
        ax.set_title(cond, fontsize=11)
        ax.set_xlabel("Time step")
        if j == 0:
            ax.set_ylabel("Target y1 (z)")
            ax.legend(loc='best', fontsize=7)
    ax_bar = fig.add_subplot(gs[1, :])
    names = list(conditions)
    vals = [effect_err[c] for c in names]
    bars = ax_bar.bar(np.arange(4), vals, 0.55,
                      color='tab:orange', edgecolor='black', lw=0.8)
    ax_bar.set_yscale('log')                       # CI dwarfs the rest on a linear axis
    ax_bar.set_ylim(0.3, 100)
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_bar.axhline(ci_limit, color='black', ls='--', lw=2,
                   label="Theoretical CI limit (no cross-variable structure)")
    ax_bar.set_xticks(np.arange(4))
    ax_bar.set_xticklabels(names, fontsize=10)
    ax_bar.set_ylabel("Effect error (MAE, °C, log)")
    ax_bar.set_title("Intervention-effect error by graph condition · sparse 10-variable system")
    ax_bar.legend()
    ax_bar.grid(True, axis='y', which='both', alpha=0.25)
    _save(fig, "example_graph_modes.png")
    plt.close(fig)


def demo3_custom_mask(model, window, graph):
    print("\n[demo 3] custom mask: in-fill a past gap AND predict the future")
    P = model.patch_size
    w = window.copy()
    w[0, CONTEXT:] = 1.0                                   # ON plan for the future
    gap_a, gap_b = CONTEXT - 320, CONTEXT - 256            # 64-step past gap
    mask = np.zeros((2, (CONTEXT + HORIZON) // P))
    mask[1, CONTEXT // P:] = 1.0                           # temp future
    mask[1, gap_a // P: gap_b // P] = 1.0                  # temp past gap
    recon = model.what_if(w, custom_mask=mask, graph=graph)[1].cpu().numpy()
    truth_fut = simulate_valve_temp(w[0], temp0=w[1, 0])[CONTEXT:]
    print(f"  gap in-fill MAE vs recorded values: {mae(recon[gap_a:gap_b], window[1, gap_a:gap_b]):.3f} degC")
    print(f"  future MAE vs physics             : {mae(recon[CONTEXT:], truth_fut):.3f} degC")

    plt = _plt()
    t_h, t_f = np.arange(-PLOT_CONTEXT, 0), np.arange(HORIZON)
    z = _zscale(window[1, :CONTEXT])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 5.8), sharex=True,
                                   height_ratios=[1.6, 1])
    obs = window[1, CONTEXT - PLOT_CONTEXT:CONTEXT].copy()
    ga, gb = gap_a - CONTEXT, gap_b - CONTEXT              # display coords
    obs[ga + PLOT_CONTEXT:gb + PLOT_CONTEXT] = np.nan      # the gap is HIDDEN
    ax1.plot(t_h, z(obs), 'k-', lw=2.2, label="History (observed)")
    ax1.axvspan(ga, gb, color='0.45', alpha=0.15, zorder=0)
    ax1.plot(np.arange(ga, gb), z(window[1, gap_a:gap_b]), color='tab:red',
             label="Ground truth (hidden)", **HALO)
    ax1.plot(np.arange(ga, gb), z(recon[gap_a:gap_b]), color='tab:red', lw=2.2,
             zorder=3, label="Prediction (masked)")
    ax1.plot(t_f, z(truth_fut), color='tab:red', **HALO)
    ax1.plot(t_f, z(recon[CONTEXT:]), color='tab:red', lw=2.2, zorder=3)
    _mark(ax1, annotate=True)
    _tag(ax1, "prediction")
    ax1.set_ylabel("Temp (z)")
    ax1.set_title("Custom mask · in-fill a hidden past gap + predict the future", pad=20)
    ax1.legend(loc='upper left', ncol=3, fontsize=8)
    ax2.step(t_h, window[0, CONTEXT - PLOT_CONTEXT:CONTEXT], where='post',
             color='black', lw=2.2, label="History")
    ax2.step(np.arange(-1, HORIZON), np.concatenate([[window[0, CONTEXT - 1]], w[0, CONTEXT:]]),
             where='post', color='tab:red', lw=2.4, label="Plan: Valve ON")
    _mark(ax2)
    _tag(ax2, "intervention")
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_ylabel("Valve [-1, 1]")
    ax2.set_xlabel("Time step")
    ax2.legend(loc='upper left', ncol=2, fontsize=8)
    _save(fig, "example_custom_mask.png")
    plt.close(fig)


def demo4_pandas(model, window, graph_named):
    print("\n[demo 4] pandas predict_df (valve plan via future_df)")
    import pandas as pd
    ts = pd.date_range("2024-01-01", periods=CONTEXT, freq="h")
    ctx_df = pd.DataFrame({"id": "plant_A", "timestamp": ts,
                           "temp": window[1, :CONTEXT], "valve": window[0, :CONTEXT]})
    fut_ts = pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
    future_df = pd.DataFrame({"id": "plant_A", "timestamp": fut_ts, "valve": 1.0})
    pred_df = model.predict_df(ctx_df, future_df=future_df, prediction_length=HORIZON,
                               quantile_levels=[0.1, 0.5, 0.9],
                               id_column="id", timestamp_column="timestamp",
                               target="temp", graph=graph_named)
    print(pred_df.head(5).to_string(index=False))
    w = window.copy()
    w[0, CONTEXT:] = 1.0
    truth = simulate_valve_temp(w[0], temp0=w[1, 0])[CONTEXT:]
    print(f"  ... final prediction {pred_df['predictions'].iloc[-1]:.2f} degC "
          f"(physics end {truth[-1]:.2f} degC)")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    hist_ts, hist_v = ts[-PLOT_CONTEXT:], window[1, CONTEXT - PLOT_CONTEXT:CONTEXT]
    ax.plot(hist_ts, hist_v, 'k-', lw=2.0, label="History")
    ax.plot(fut_ts, truth, color='tab:red', label="Ground truth (physics)", **HALO)
    ax.fill_between(fut_ts, pred_df["0.1"], pred_df["0.9"], color='tab:red',
                    alpha=0.18, label="10–90% band")
    ax.plot(fut_ts, pred_df["predictions"], color='tab:red', lw=2.2, zorder=3,
            label="predictions (median)")
    ax.axvline(ts[-1], color='0.15', lw=1.3, ls=(0, (4, 2.5)))
    ax.grid(True)
    ax.set_ylabel("Temp (°C)")
    ax.set_title("predict_df · do(valve := ON) supplied via future_df", pad=8)
    ax.legend(loc='upper left', ncol=4, fontsize=8)
    _save(fig, "example_predict_df.png")
    plt.close(fig)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "software-competence-center-hagenberg/PCFM"
    model = PCFMPredictor.from_pretrained(ckpt)

    graph = [(0, 1, float(LAG))]        # valve -> temp with the true delay
    window = make_valve_temp_window()

    demo1_whatif(model, window, graph)
    demo2_graph_modes(model)
    demo3_custom_mask(model, window, graph)
    demo4_pandas(model, window, graph_named=[("valve", "temp", float(LAG))])
    demo5_forecast(model, window, graph)


if __name__ == "__main__":
    main()
