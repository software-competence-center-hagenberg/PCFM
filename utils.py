"""PCFM utils: a training-free test-time recipe on top of model.py (the GIFT-Eval submission recipe).

    from utils import load_model, PCFMForecaster
    model = load_model()                                            # huggingface.co/software-competence-center-hagenberg/PCFM
    fc = PCFMForecaster(model, prediction_length=48, freq="H")     # season_length from freq (24); both optional
    q = fc.predict([y1, y2])                                        # each [T] or [C, T] (NaNs allowed) -> [9, H] / [9, H, C]
    PCFMForecaster(model, 48, ensemble=False).predict([y1])        # one plain forward pass, no recipe

freq / season_length only switch the frequency-dependent rules: without freq there is no context cap, monthly series
are stretched like the rest and the flip is never skipped; without a seasonality (freq unknown) dilation never fires.

Channel-independent: every channel is forecast on its own; nothing touches the weights.

Inference recipe (DEPLOY below). Per series, six members are averaged:
    sign flip {off, on}  x  dilation {1, 2, 4}
  * sign flip  : forecast -x and mirror the quantiles back (RevIN's one free symmetry); skipped on univariate
                 sub-daily series (S / T / H).
  * dilation d : block-average the context by d, forecast ceil(H/d) coarse steps, interpolate back. Only on series
                 longer than the model window (8192) whose coarse context still fills >= half of it, and only when
                 d divides the seasonality; otherwise the member equals the d = 1 member. On horizons >= 12 patches
                 the dilated members REPLACE the native ones unless the increments are smooth (diff-ACF > 0.5).
  * horizon pad: 64 extra steps are forecast and cut off (the last patch of a block is the weakest).
  * context cap: 2048 steps on H / D / W / M data (never below 8 horizons).
  * stretch    : contexts shorter than 8 patches (16 on monthly data) are linearly up-sampled by s <= 4, forecast
                 on the stretched grid, mapped back, and averaged with the native forecast.
  * leading NaNs are trimmed, interior NaNs forward-filled; integer-valued series get integer quantiles.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from model import GraphInput, HF_REPO_ID, PCFMPredictor

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEPLOY = dict(
    sign_flip=(False, True), flip_skip_univariate_subdaily=True,
    dilations=(1, 2, 4), dilation_min_fill=0.5, dilation_replace_min_horizon_patches=12, dilation_replace_max_diff_acf=0.5,
    horizon_pad=64,
    context_cap_by_freq={"D": 2048, "W": 2048, "M": 2048, "H": 2048}, context_cap_min_horizons=8,
    stretch_min_patches=8, stretch_min_patches_by_freq={"M": 16}, stretch_max_factor=4, stretch_max_horizon=2048,
    round_integer=True,
)
SPEED = dict(token_budget=10_000_000, min_batch=4, max_batch=8192, chunk=1024)   # batch = token budget / (C x (ctx + H))


# ------------------------------------------------------------------------------------------------- model
def load_model(checkpoint=HF_REPO_ID, device=None):
    return PCFMPredictor.from_pretrained(checkpoint, device=device, runtime_discrete_revin=False).model   # as evaluated


# ------------------------------------------------------------------------------------------------- series helpers
def _ffill(a):
    """[C, T]: forward fill (leading NaNs <- first observation); rows without any observation stay NaN."""
    a = a.copy()
    for c in range(a.shape[0]):
        nan = np.isnan(a[c])
        obs = np.where(~nan)[0]
        if 0 < obs.size < a.shape[1]:
            pos = np.maximum.accumulate(np.where(nan, -1, np.arange(a.shape[1])))
            a[c] = a[c, np.where(pos < 0, obs[0], pos)]
    return a


def _diff_acf1(x, max_len=4096):
    """Lag-1 autocorrelation of the first differences (max over channels; 1.0 if undefined): ~-0.5 noise, ~+0.9 smooth."""
    out = -np.inf
    for row in np.atleast_2d(np.asarray(x, dtype=np.float64))[:, -max_len:]:
        r = row[np.isfinite(row)]
        if r.size < 32:
            continue
        d = np.diff(r)
        d = d - d.mean()
        den = float((d * d).sum())
        out = max(out, float((d[1:] * d[:-1]).sum()) / den if den > 0 else 1.0)
    return 1.0 if not np.isfinite(out) else out


def _dilate(a, d):
    """[C, T] -> block means of d steps (oldest remainder dropped), NaN-aware."""
    n = a.shape[1] // d
    with np.errstate(invalid="ignore"):
        return np.nanmean(a[:, a.shape[1] - n * d:].reshape(a.shape[0], n, d), axis=-1)


def _undilate(q, d, H):
    """[Q, Hc, C] on the coarse grid -> [Q, H, C] by linear interpolation at the original steps."""
    f = np.clip((np.arange(1, H + 1) - 0.5) / d - 0.5, 0, q.shape[1] - 1)
    lo = np.floor(f).astype(int)
    hi = np.clip(lo + 1, 0, q.shape[1] - 1)
    w = (f - lo).astype(q.dtype)[None, :, None]
    return (1.0 - w) * q[:, lo] + w * q[:, hi]


def _stretch(a, s):
    """[C, T] -> [C, T*s], linear (align_corners=False, as in the training augmentation); NaNs stay NaN."""
    nan = np.isnan(a)
    f = np.array([np.interp(np.arange(a.shape[1]), np.where(~n)[0], r[~n]) if (~n).any() else np.zeros_like(r)
                  for r, n in zip(a, nan)], dtype=np.float32)
    y = F.interpolate(torch.from_numpy(f)[None], size=a.shape[1] * s, mode="linear", align_corners=False)[0].numpy()
    if nan.any():
        m = F.interpolate(torch.from_numpy(nan.astype(np.float32))[None], size=a.shape[1] * s, mode="nearest")[0].numpy()
        y = np.where(m > 0.5, np.nan, y)
    return y


def _unstretch(q, s, H):
    """[Q, H*s, C] -> [Q, H, C]: original step h sits at stretched index s*h - (s+1)/2."""
    f = s * np.arange(1, H + 1, dtype=np.float64) - (s + 1) / 2.0
    lo = np.clip(np.floor(f).astype(int), 0, q.shape[1] - 1)
    hi = np.clip(lo + 1, 0, q.shape[1] - 1)
    w = (f - lo).astype(q.dtype)[None, :, None]
    return (1.0 - w) * q[:, lo] + w * q[:, hi]


def _freq_class(freq):
    """'5T'/'15min' -> 'T', 'H' -> 'H', 'W-SUN' -> 'W', 'MS' -> 'M', '10S' -> 'S', 'Q*' -> 'Q', 'A*'/'Y*' -> 'Y'."""
    if freq is None:
        return ""
    f = "".join(ch for ch in str(getattr(freq, "freqstr", freq)) if not ch.isdigit()).lstrip("-").upper()
    if f.startswith(("MIN", "T")):
        return "T"
    for c, keys in (("H", "H"), ("D", ("D", "B")), ("W", "W"), ("M", "M"), ("Q", "Q"), ("Y", ("A", "Y")), ("S", "S")):
        if f.startswith(keys):
            return c
    return f[:1]


_SEASON = {"S": 3600, "T": 1440, "H": 24, "D": 1, "B": 5, "W": 1, "M": 12, "Q": 4, "Y": 1}   # per unit (as GluonTS)


def season_from_freq(freq):
    """Seasonality in steps from a pandas frequency string ('5T' -> 288, 'H' -> 24, 'M' -> 12); 1 if unknown."""
    if freq is None:
        return 1
    f = str(getattr(freq, "freqstr", freq)).split("-")[0].upper().replace("MIN", "T").replace("ME", "M").replace("MS", "M")
    mult = int("".join(ch for ch in f if ch.isdigit()) or 1)
    base = _SEASON.get("".join(ch for ch in f if not ch.isdigit())[:1], 1)
    return base // mult if base % mult == 0 else 1


# ------------------------------------------------------------------------------------------------- predictor
class PCFMForecaster:
    """PCFM with the inference recipe for one horizon (and, optionally, frequency / seasonality).
    predict(): list of [T] or [C, T] arrays -> list of quantile arrays [Q, H] or [Q, H, C] (levels QUANTILES).
      ensemble=False  -> one plain forward pass per series (NaNs mean-filled), no recipe at all
      freq=None       -> the recipe without its frequency-dependent rules (no cap, no flip gate, 8-patch stretch)
      season_length   -> default season_from_freq(freq); 1 (no dilation) when freq is unknown"""
    def __init__(self, model, prediction_length, freq=None, season_length=None, ensemble=True, cfg=DEPLOY,
                 batch_size=None):
        self.model, self.H, self.cfg, self.batch_size = model, int(prediction_length), cfg, batch_size
        self.ensemble = bool(ensemble)
        self.device = next(model.parameters()).device
        self.P = model.patch_size
        self.max_ctx = model.backbone.max_context_steps * model.max_unrolled_chunks
        self.fc = _freq_class(freq)
        self.season = int(season_length) if season_length else season_from_freq(freq)

    # ---- one forward pass per batch of series ([C, T] arrays, NaNs allowed) -> [Q, L, C] each
    @torch.no_grad()
    def _forward(self, series, signs, L):
        P, out = self.P, []
        for i in range(0, len(series), self.batch_size):
            ser = [s.copy() for s in series[i:i + self.batch_size]]
            sg = signs[i:i + self.batch_size]
            lens = [math.ceil(min(self.max_ctx, s.shape[1]) / P) * P for s in ser]
            C = [s.shape[0] for s in ser]
            offsets = [np.zeros((len(QUANTILES), c)) for c in C]
            chunks, rem = [], L
            while rem > 0:
                step = min(SPEED["chunk"], rem)
                q = self._run(ser, sg, lens, math.ceil(step / P) * P)[:, :step]          # [nodes, step, Q]
                k, part = 0, []
                for j, c in enumerate(C):
                    qj = q[k:k + c]
                    k += c
                    qq = np.ascontiguousarray(qj.transpose(2, 1, 0)) + offsets[j][:, None, :]
                    part.append(qq)
                    if rem > step:                                    # next chunk: median appended, band carried
                        offsets[j] = qq[:, -1, :] - qq[4, -1, :]
                        ser[j] = np.concatenate([ser[j], qj[:, :, 4]], axis=1)
                chunks.append(part)
                rem -= step
            out += [np.sort(np.concatenate([ch[j] for ch in chunks], axis=1), axis=0) for j in range(len(ser))]
        return out

    def _run(self, ser, signs, lens, pred_len):
        Tmax = max(lens)
        rows, sgn = [], []
        for s, sg, n in zip(ser, signs, lens):
            for r in s:
                r = np.where(np.isnan(r), np.nanmean(r) if np.isfinite(r).any() else 0.0, r)
                rows.append((sg * r[-n:]).astype(np.float32))
                sgn.append(sg)
        N = len(rows)
        X, pad = np.zeros((N, Tmax), np.float32), np.zeros((N, Tmax), np.float32)
        for k, r in enumerate(rows):                                          # left-padded to the longest in the batch
            X[k, Tmax - len(r):] = r
            pad[k, :Tmax - len(r)] = 1.0                                      # m_pad_step: masked by the model
        dev = self.device
        data = GraphInput(x=torch.from_numpy(X).to(dev), edge_index=torch.arange(N, device=dev).repeat(2, 1),
                          edge_attr=torch.zeros(N, 1, device=dev), batch=torch.arange(N, device=dev), num_graphs=N,
                          m_pad_step=torch.from_numpy(pad).unsqueeze(-1).to(dev))
        q = self.model(data, mode="forecast", pred_len=pred_len)
        neg = torch.tensor(sgn, device=dev).view(-1, 1, 1) < 0
        return torch.where(neg, -q.flip(-1), q).float().cpu().numpy()      # mirror flipped series back

    # ---- context stretching on top of a forward: [C, T] arrays -> [Q, L, C]
    def _stretch_factor(self, T, L):
        need = self.cfg["stretch_min_patches_by_freq"].get(self.fc, self.cfg["stretch_min_patches"]) * self.P
        if T < 2 or T >= need:
            return 1
        s = max(1, min(math.ceil(need / T), self.cfg["stretch_max_factor"]))
        if s * L > self.cfg["stretch_max_horizon"]:
            return 1
        while s > 1 and s * T > self.max_ctx:
            s -= 1
        return s

    def _forecast(self, series, signs, L):
        out = [None] * len(series)
        groups = {}
        for k, s in enumerate(series):
            groups.setdefault(self._stretch_factor(s.shape[1], L), []).append(k)
        for s in sorted(groups):
            idx = groups[s]
            native = self._forward([series[k] for k in idx], [signs[k] for k in idx], L)
            if s > 1:
                stretched = self._forward([_stretch(series[k], s) for k in idx], [signs[k] for k in idx], L * s)
                native = [np.sort(0.5 * (_unstretch(qs, s, L) + qn[:, :L]), axis=0) for qs, qn in zip(stretched, native)]
            for k, q in zip(idx, native):
                out[k] = q
        return out

    # ---- one member (flip, d) of one series: the transformed context and its effective settings
    def _member(self, t, flip, d, univariate):
        cfg, T = self.cfg, t.shape[1]
        cap = cfg["context_cap_by_freq"].get(self.fc)
        L = T if not cap else min(T, max(cap, math.ceil(cfg["context_cap_min_horizons"] * self.H / self.P) * self.P))
        x = t[:, T - L:]
        need = max(self.max_ctx + 1, 2 * d, math.ceil(cfg["dilation_min_fill"] * d * self.max_ctx))
        d = d if (d > 1 and self.season % d == 0 and x.shape[1] >= need) else 1
        fin = np.isfinite(x).any(axis=0)
        x = _ffill(x[:, int(np.argmax(fin)) if fin.any() else 0:])                  # leading NaNs trimmed, rest ffilled
        if flip and cfg["flip_skip_univariate_subdaily"] and self.fc in ("S", "T", "H") and univariate:
            flip = False
        return (_dilate(x, d) if d > 1 else x), flip, d

    def auto_batch_size(self, series):
        """Series per forward from a token budget: series x channels x (context + horizon) <= SPEED["token_budget"]."""
        tok = [(1 if np.ndim(t) == 1 else np.shape(t)[0]) * (min(np.shape(t)[-1], self.max_ctx) + self.H) for t in series[:512]]
        bs = int(SPEED["token_budget"] / max(float(np.percentile(tok, 90)), 1.0))
        return max(SPEED["min_batch"], min(SPEED["max_batch"], bs))

    def predict(self, series):
        H, cfg = self.H, self.cfg
        if self.batch_size is None:
            self.batch_size = self.auto_batch_size(series)
        targets = [np.atleast_2d(np.asarray(t, dtype=np.float32)) for t in series]
        if not self.ensemble:                                                    # plain: one forward, horizon H
            return [q[..., 0] if np.ndim(y) == 1 else q
                    for y, q in zip(series, self._forward(targets, [1.0] * len(targets), H))]
        members = [(f, d) for f in cfg["sign_flip"] for d in cfg["dilations"]]
        jobs, alias, mem = {}, {}, [dict() for _ in series]
        for vi, (f, d) in enumerate(members):                                    # identical members run once
            for i, t in enumerate(targets):
                x, f_, d_ = self._member(t, f, d, np.ndim(series[i]) == 1)
                key = (i, d_, f_, x.shape[1])
                if key in alias:
                    mem[i][vi] = alias[key]
                    continue
                alias[key] = (i, vi)
                jobs.setdefault(d_, []).append((i, vi, x, f_))
        dil_of = {}
        for d, items in jobs.items():
            Hd = math.ceil(H / d) if d > 1 else H
            qs = self._forecast([it[2] for it in items], [-1.0 if it[3] else 1.0 for it in items], Hd + cfg["horizon_pad"])
            for (i, vi, _, _), q in zip(items, qs):
                mem[i][vi] = _undilate(q[:, :Hd], d, H) if d > 1 else q[:, :H]
                dil_of[(i, vi)] = d
        replace = math.ceil(H / self.P) >= cfg["dilation_replace_min_horizon_patches"]
        out = []
        for i, y in enumerate(series):
            m = {vi: (mem[i][q[1]] if isinstance(q, tuple) else q) for vi, q in mem[i].items()}
            d_vi = {vi: dil_of[q if isinstance(q, tuple) else (i, vi)] for vi, q in mem[i].items()}
            vis = sorted(m)
            dil = [vi for vi in vis if d_vi[vi] > 1]
            if replace and dil and len(dil) < len(vis) and _diff_acf1(y) <= cfg["dilation_replace_max_diff_acf"]:
                vis = dil                                                            # dilated members replace native
            q = m[vis[0]] if len(vis) == 1 else sum((1.0 / len(vis)) * m[vi] for vi in vis)
            fin = targets[i][np.isfinite(targets[i])]
            if cfg["round_integer"] and fin.size and np.abs(fin - np.round(fin)).max() < 1e-6 and np.ptp(fin) > 0:
                q = np.round(q)
            q = np.sort(q, axis=0)
            out.append(q[..., 0] if np.ndim(y) == 1 else q)
        return out
