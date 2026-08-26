"""PCFM — Pre-trained Causal Foundation Model for multivariate time series.

Self-contained inference code for the checkpoint published at
    https://huggingface.co/software-competence-center-hagenberg/PCFM

The architecture below is copied verbatim from the training / evaluation code,
with every configuration value of the published checkpoint hardcoded, so this
single file plus the weights is everything needed for inference.

Dependencies: torch >= 2.4 (nn.RMSNorm), numpy, and (only for downloading the
checkpoint from the Hub) huggingface_hub.

Public API (see README.md for details):

    from model import PCFMPredictor

    model = PCFMPredictor.from_pretrained()            # downloads from the Hub
    med   = model.forecast(history, prediction_length=64)
    recon = model.what_if(window, context_length=1024, masked_nodes=[0])

`history` / `window` are [num_nodes, time] tensors of one multivariate system.
A causal graph (`edge_index`, optionally `edge_lags` in time steps) can be
supplied; without one the model falls back to a fully connected graph with
zero lags. `what_if` runs masked reconstruction: the futures of the nodes in
`masked_nodes` are hidden and re-predicted while every other node's future
(e.g. an intervention plan on an actuator) stays observed — the do-operator
style counterfactual interface. A fully custom patch-level mask can be passed
instead via `custom_mask`.
"""
import math
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# HARDCODED CONFIGURATION OF THE PUBLISHED CHECKPOINT
# =========================================================================
CONFIG = {
    "dim": 512,                  # model width
    "patch_size": 32,            # temporal patch length (all masks are patch-aligned)
    "layers": 11,                # transformer blocks
    "num_heads": 8,
    "max_context_steps": 8192,   # longest context the backbone attends over
    "pred_len": 128,             # default forecast horizon (any length works)
    "use_layer_scale": False,
    "enable_dedicated_spatial": False,
    "spatial_stride": 1,         # spatial (graph) mixer in every eligible block
    "max_unrolled_chunks": 1,
    "enable_dense_summary": True,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # RevIN behaviour (matches pretraining):
    "enable_runtime_discrete_revin": False,
    "discrete_max_states": 20,
}
HF_REPO_ID = "software-competence-center-hagenberg/PCFM"


# =========================================================================
# GRAPH INPUT CONTAINER (plain attributes -- no torch_geometric dependency)
# =========================================================================
class GraphInput:
    """Minimal attribute container the backbone reads its inputs from."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def fully_connected_graph(num_nodes: int, device=None) -> torch.Tensor:
    """[2, E] edge index of all directed pairs incl. self-loops (the fallback
    topology when no causal graph is supplied)."""
    idx = torch.arange(num_nodes, device=device)
    src = idx.repeat_interleave(num_nodes)
    dst = idx.repeat(num_nodes)
    return torch.stack([src, dst], dim=0)


# =========================================================================
# ARCHITECTURE (verbatim from the training / evaluation code)
# =========================================================================
def apply_rope(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class FusedSwiGLU(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        hidden_dim = 256 * (((int(2 * dim * 4 / 3)) + 256 - 1) // 256)
        self.w_gate_val = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, val = self.w_gate_val(x).chunk(2, dim=-1)
        return self.w_out(self.dropout(F.silu(gate) * val))


class SwiGLUResidualBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout_p=0.0):
        super().__init__()
        self.gate_val = nn.Linear(in_dim, 2 * h_dim, bias=False)
        self.output_layer = nn.Linear(h_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        gate, val = self.gate_val(x).chunk(2, dim=-1)
        return self.dropout(self.output_layer(F.silu(gate) * val)) + self.residual_layer(x)


class DualSwiGLUProjector(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout_p=0.0):
        super().__init__()
        self.block1 = SwiGLUResidualBlock(in_dim, h_dim, out_dim, dropout_p)
        self.norm = nn.RMSNorm(out_dim)
        self.block2 = SwiGLUResidualBlock(out_dim, h_dim, out_dim, dropout_p)

    def forward(self, x):
        x = self.block1(x)
        x = self.norm(x).to(x.dtype)
        return self.block2(x)


class PatchRobustRevIN(nn.Module):
    """Instance normalization with two tracks (parity with pretraining):
    continuous series -> asinh z-score with masked statistics; discrete
    (low-cardinality) series -> per-window min-max to [-1, 1]. Runtime
    discreteness detection routes windows with <= discrete_max_states
    tolerance-distinct levels to the discrete track when enabled."""

    def __init__(self, num_features, eps=1e-6, detect_discrete_runtime=False,
                 discrete_max_states=20):
        super().__init__()
        self.eps = eps
        self.detect_discrete_runtime = detect_discrete_runtime
        self.discrete_max_states = discrete_max_states

    @torch.no_grad()
    def _detect_discrete(self, x_float, mask=None, min_valid=8):
        N, C, T = x_float.shape
        xs = x_float.reshape(N * C, T)
        v = (mask.reshape(N * C, T) < 0.5) if mask is not None else torch.ones_like(xs, dtype=torch.bool)
        has_v = v.any(dim=1)
        first_idx = torch.argmax(v.float(), dim=1).clamp_max(T - 1)
        fill = xs.gather(1, first_idx.unsqueeze(1))
        xs_f = torch.where(v, xs, fill)
        s, _ = xs_f.sort(dim=1)
        rng_row = s[:, -1] - s[:, 0]
        tol = (rng_row * 1e-4).clamp_min(1e-8).unsqueeze(1)
        n_eff = 1 + ((s[:, 1:] - s[:, :-1]) > tol).sum(dim=1)
        is_disc = (n_eff <= self.discrete_max_states) & (v.sum(dim=1) >= min_valid) \
                  & has_v & (rng_row > self.eps)
        return is_disc.view(N, C).all(dim=1)

    def _get_patch_statistics(self, x_float, mask=None):
        if mask is not None:
            valid = (~mask.bool()).to(x_float.dtype)
            count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_val = (x_float * valid).sum(dim=-1, keepdim=True) / count
            var = (((x_float - mean_val) * valid) ** 2).sum(dim=-1, keepdim=True) / count
        else:
            mean_val = x_float.mean(dim=-1, keepdim=True)
            var = x_float.var(dim=-1, keepdim=True, unbiased=False)
        scale = torch.sqrt(var)
        return (torch.nan_to_num(mean_val, nan=0.0),
                torch.nan_to_num(scale, nan=1.0).clamp_min(self.eps))

    def forward(self, x, mode, mask=None, stats=None, is_discrete=None):
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type if x.device.type in ["cuda", "cpu"] else "cuda", enabled=False):
            x_float = x.to(torch.float32)
            if mode == 'norm':
                if self.detect_discrete_runtime:
                    is_discrete = self._detect_discrete(x_float, mask)
                with torch.no_grad():
                    mean, scale = self._get_patch_statistics(x_float, mask)
                    s_min = torch.amin(x_float, dim=-1, keepdim=True)
                    s_max = torch.amax(x_float, dim=-1, keepdim=True)
                    s_range = s_max - s_min
                    is_dynamic = (s_range > self.eps)
                denom = torch.where(is_dynamic, s_range, torch.ones_like(s_range))
                x_discrete = torch.where(is_dynamic, 2.0 * ((x_float - s_min) / denom) - 1.0,
                                         x_float - s_min)
                x_continuous = torch.asinh((x_float - mean) / scale)
                if is_discrete is not None:
                    route_mask = is_discrete.view(-1, 1, 1).to(torch.bool)
                    x_norm = torch.where(route_mask, x_discrete, x_continuous)
                else:
                    x_norm = x_continuous
                    route_mask = torch.zeros((x_float.size(0), 1, 1), dtype=torch.bool, device=x.device)
                return x_norm.to(orig_dtype), (mean, scale, s_min, s_max, route_mask, is_dynamic)
            elif mode == 'denorm':
                mean, scale, s_min, s_max, route_mask, is_dynamic = stats
                x_denorm_discrete = torch.where(is_dynamic,
                                                ((x_float + 1.0) / 2.0) * (s_max - s_min) + s_min,
                                                x_float + s_min)
                x_denorm_continuous = torch.sinh(torch.clamp(x_float, min=-20.0, max=20.0)) * scale + mean
                return torch.where(route_mask, x_denorm_discrete, x_denorm_continuous).to(orig_dtype)


class StandardRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 256, theta: float = 10000.0):
        super().__init__()
        self.head_dim, self.max_seq_len = head_dim, max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        return (self.cos_cached[position_ids].unsqueeze(2).to(dtype),
                self.sin_cached[position_ids].unsqueeze(2).to(dtype))


class HeadwiseMultiScaleRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, num_heads: int, max_seq_len: int = 256,
                 min_theta: float = 100.0, max_theta: float = 20000.0):
        super().__init__()
        self.head_dim, self.num_heads, self.max_seq_len = head_dim, num_heads, max_seq_len
        cos_list, sin_list = [], []
        for h in range(num_heads):
            theta_h = min_theta * (max_theta / min_theta) ** (h / max(1, num_heads - 1))
            inv_freq = 1.0 / (theta_h ** (torch.arange(0, head_dim, 2).float() / head_dim))
            freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_list.append(emb.cos().unsqueeze(1))
            sin_list.append(emb.sin().unsqueeze(1))
        self.register_buffer("cos_cached", torch.cat(cos_list, dim=1), persistent=False)
        self.register_buffer("sin_cached", torch.cat(sin_list, dim=1), persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        return self.cos_cached[position_ids].to(dtype), self.sin_cached[position_ids].to(dtype)


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0,
                 max_seq_len: int = 256, window_size: int = None, use_headwise_rope: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop_p = dropout
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.rope = (HeadwiseMultiScaleRotaryEmbedding(self.head_dim, num_heads, max_seq_len)
                     if use_headwise_rope else StandardRotaryEmbedding(self.head_dim, max_seq_len))

    def forward(self, x: torch.Tensor, is_causal: bool = True, patch_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q = self.q_norm(qkv[:, :, 0]).to(qkv.dtype)
        k = self.k_norm(qkv[:, :, 1]).to(qkv.dtype)
        v = qkv[:, :, 2].transpose(1, 2).contiguous()

        cos, sin = self.rope(torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
                             + (self.rope.max_seq_len - N), dtype=q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        q, k = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()

        valid_mask = ((patch_mask.squeeze(-1) < 0.5).view(B, 1, 1, N) if patch_mask is not None
                      else torch.ones((B, 1, 1, N), dtype=torch.bool, device=x.device))
        is_self = (torch.arange(N, device=x.device).unsqueeze(1)
                   == torch.arange(N, device=x.device).unsqueeze(0)).view(1, 1, N, N)
        causal_mask = (torch.tril(torch.ones(N, N, dtype=torch.bool, device=x.device)).view(1, 1, N, N)
                       if is_causal else torch.ones((1, 1, N, N), dtype=torch.bool, device=x.device))
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=(valid_mask & causal_mask) | is_self,
            dropout_p=self.attn_drop_p if self.training else 0.0, scale=self.scale)
        return self.proj(attn_out.transpose(1, 2).contiguous().reshape(B, N, C))


class SpatialCausalLagMixer(nn.Module):
    """Graph message passing across nodes with per-edge transport delays: each
    directed edge (src -> dst, lag) lets dst attend to src's representation
    `lag` time steps in the past ('lagged' tap) or to src's final summary patch
    ('summary' tap, used for chunk-unrolling edges)."""

    def __init__(self, dim: int, num_heads: int = 8, patch_size: int = 32,
                 dropout: float = 0.0, use_gate_norm: bool = False,
                 use_gate_global: bool = False, chunk_size: Optional[int] = 1000):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.patch_size = float(patch_size)
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5
        self.taps = ['lagged', 'summary']
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.use_gate_norm, self.use_gate_global = use_gate_norm, use_gate_global
        self.msg_dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_lags, patch_mask=None, x0=None, edge_type=None):
        src, dst = edge_index[0], edge_index[1]
        E = src.numel()
        if E == 0:
            return torch.zeros_like(x)
        N, P, D = x.shape
        H, d = self.num_heads, self.head_dim

        m_valid = ((patch_mask < 0.5).to(x.dtype).view(N, P, 1, 1) if patch_mask is not None
                   else torch.ones((N, P, 1, 1), device=x.device, dtype=x.dtype))

        qkv = self.qkv(x).view(N, P, 3, H, d)
        q = self.q_norm(qkv[:, :, 0]).to(qkv.dtype)
        k = self.k_norm(qkv[:, :, 1]).to(qkv.dtype) * m_valid
        v = qkv[:, :, 2] * m_valid
        k_flat, v_flat = k.view(N * P, H, d), v.view(N * P, H, d)
        m_flat = m_valid.view(N * P, 1, 1)

        agg_spatial = torch.zeros((N, P, H, d), device=x.device, dtype=torch.float32)
        agg_summary = torch.zeros((N, P, H, d), device=x.device, dtype=torch.float32)
        deg_spatial = torch.zeros(N, device=x.device, dtype=torch.float32)
        deg_summary = torch.zeros(N, device=x.device, dtype=torch.float32)
        grid = torch.arange(P, device=x.device, dtype=x.dtype).view(1, P)
        if edge_type is None:
            edge_type = torch.zeros((E,), dtype=torch.long, device=x.device)

        use_chunking = (self.chunk_size is not None) and (E > self.chunk_size)
        step = self.chunk_size if use_chunking else E

        for start_idx in range(0, E, step):
            end_idx = min(start_idx + step, E)
            c_src, c_dst = src[start_idx:end_idx], dst[start_idx:end_idx]
            c_edge_lags = edge_lags[start_idx:end_idx]
            c_edge_type = edge_type[start_idx:end_idx]
            c_E = c_src.numel()

            is_self_loop = (c_src == c_dst).view(c_E, 1, 1, 1)
            is_cross_node = (~is_self_loop).to(x.dtype)
            is_summary_flat = (c_edge_type == 1)
            is_spatial_flat = is_cross_node.view(-1).bool() & (~is_summary_flat)
            is_summary_edge = is_summary_flat.view(c_E, 1, 1, 1)

            q_dst = q[c_dst]
            lags_float = F.relu(c_edge_lags / self.patch_size).clamp(min=1e-4).view(-1, 1)
            idx_peak_lagged = (grid - lags_float).clamp(min=0.0).long()
            flat_idx_lagged = c_src.unsqueeze(1) * P + idx_peak_lagged

            tap_msgs = []
            for tap in self.taps:
                if tap == 'lagged':
                    k_tap, v_tap = k_flat[flat_idx_lagged], v_flat[flat_idx_lagged]
                    m_lag = m_flat[flat_idx_lagged.view(-1)].view(c_E, P, 1, 1)
                    is_invalid_tap = (m_lag < 0.5).expand(c_E, P, H, 1) | is_summary_edge
                elif tap == 'summary':
                    idx_summary = torch.full((c_E, P), P - 1, dtype=torch.long, device=x.device)
                    flat_idx_summary = c_src.unsqueeze(1) * P + idx_summary
                    k_tap, v_tap = k_flat[flat_idx_summary], v_flat[flat_idx_summary]
                    m_summary = m_flat[flat_idx_summary.view(-1)].view(c_E, P, 1, 1)
                    is_invalid_tap = (m_summary < 0.5).expand(c_E, P, H, 1) | (~is_summary_edge)
                val_mask = (~is_invalid_tap).to(x.dtype)
                compat = q_dst * k_tap
                tap_msgs.append(F.silu(compat) * v_tap * val_mask)

            msgs = sum(tap_msgs) * is_cross_node
            msgs = self.msg_dropout(msgs)
            msgs_spatial = msgs * is_spatial_flat.view(c_E, 1, 1, 1)
            msgs_summary = msgs * is_summary_edge

            deg_spatial.index_add_(0, c_dst, is_spatial_flat.float())
            deg_summary.index_add_(0, c_dst, is_summary_flat.float())
            agg_spatial.index_add_(0, c_dst, msgs_spatial.float())
            agg_summary.index_add_(0, c_dst, msgs_summary.float())

        agg_spatial = agg_spatial / torch.sqrt(torch.clamp(deg_spatial, min=1.0).view(N, 1, 1, 1))
        agg_summary = agg_summary / torch.sqrt(torch.clamp(deg_summary, min=1.0).view(N, 1, 1, 1))
        agg = (agg_spatial + agg_summary).to(x.dtype)
        return self.proj(agg.reshape(N, P, D).to(self.proj.weight.dtype))


class PCFMBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0, ls_eps=0.0001, use_layer_scale=False,
                 scale_ffn=False, drop_path: float = 0.0, max_seq_len: int = 64,
                 spatial_mixer: nn.Module = None, dedicated_spatial: bool = False):
        super().__init__()
        self.use_layer_scale, self.scale_ffn, self.dedicated_spatial = use_layer_scale, scale_ffn, dedicated_spatial
        self.resid_dropout = nn.Dropout(dropout)
        self.mixer_norm = nn.RMSNorm(dim)
        self.spatial_attn = spatial_mixer
        self.temp_attn = (None if (dedicated_spatial and spatial_mixer is not None)
                          else TemporalSelfAttention(dim, num_heads, dropout=dropout, max_seq_len=max_seq_len))
        self.norm_ffn = nn.RMSNorm(dim)
        self.ffn = FusedSwiGLU(dim, dropout=dropout)
        if use_layer_scale:
            self.ls_mix = nn.Parameter(torch.ones(dim) * ls_eps)
            self.ls_ffn = nn.Parameter(torch.ones(dim) * ls_eps) if scale_ffn else None

    def forward(self, x, edge_index, edge_attr, patch_mask=None, batch=None, num_graphs=None,
                x0=None, has_neighbors=None, edge_type=None):
        residual = x
        x_norm = self.mixer_norm(x).to(x.dtype)
        if self.temp_attn is None:
            h_mix = self.spatial_attn(x_norm, edge_index, edge_attr, patch_mask=patch_mask, edge_type=edge_type)
        elif self.spatial_attn is None:
            h_mix = self.temp_attn(x_norm, is_causal=False, patch_mask=patch_mask)
        else:
            h_mix = (self.temp_attn(x_norm, is_causal=False, patch_mask=patch_mask)
                     + self.spatial_attn(x_norm, edge_index, edge_attr, patch_mask=patch_mask, edge_type=edge_type))
        h_mix = self.resid_dropout(h_mix)
        if self.use_layer_scale:
            h_mix = self.ls_mix.view(1, 1, -1) * h_mix
        residual = residual + h_mix
        h_ffn = self.resid_dropout(self.ffn(self.norm_ffn(residual).to(residual.dtype)))
        if self.use_layer_scale and self.ls_ffn is not None:
            h_ffn = self.ls_ffn.view(1, 1, -1) * h_ffn
        return residual + h_ffn


class PatchEmbedderMAE(nn.Module):
    def __init__(self, out_dim=256, patch_size=32, max_context_steps=8192, enable_multi_scale=False):
        super().__init__()
        self.patch_size, self.out_dim = patch_size, out_dim
        self.enable_multi_scale = enable_multi_scale
        self.max_patches = max_context_steps // patch_size
        num_input_channels = 3   # signal value + observation indicator + time ramp
        h_dim = 256 * (((int(2 * out_dim * 4 / 3)) + 256 - 1) // 256)
        self.projector = DualSwiGLUProjector(in_dim=patch_size * num_input_channels,
                                             h_dim=h_dim, out_dim=out_dim, dropout_p=0.0)

    def forward(self, x, data, mask_step=None, context_len=None, training_mode='mae'):
        N, _, L = x.shape
        P = self.patch_size
        num_patches = L // P
        obs_indicator = (1.0 - mask_step) if mask_step is not None else torch.ones_like(x)
        x_masked = x * obs_indicator
        x_base = x_masked.view(N, num_patches, P)
        obs_base = obs_indicator.view(N, num_patches, P)
        pos_ids = torch.arange(num_patches, device=x.device, dtype=x.dtype) + (self.max_patches - num_patches)
        ramp = (pos_ids + 1.0) / self.max_patches
        ramp_base = ramp.view(1, num_patches, 1).expand(N, num_patches, P)
        return self.projector(torch.cat([x_base, obs_base, ramp_base], dim=-1))


class PCFM(nn.Module):
    def __init__(self, dim=512, patch_size=32, layers=11, num_heads=8, max_context_steps=8192,
                 enable_depth_scaling: bool = False, use_layer_scale: bool = False,
                 enable_dedicated_spatial: bool = False, enable_multi_scale: bool = False,
                 spatial_stride: int = 1):
        super().__init__()
        self.dim, self.patch_size, self.layers, self.max_context_steps = dim, patch_size, layers, max_context_steps
        self.embed = PatchEmbedderMAE(dim, patch_size, max_context_steps, enable_multi_scale)
        self.blocks = nn.ModuleList()
        self.spatial_layer_indices = list(range(1, layers, spatial_stride))
        if (layers - 1) in self.spatial_layer_indices:
            self.spatial_layer_indices.remove(layers - 1)
        for i in range(layers):
            s_mixer = SpatialCausalLagMixer(dim, num_heads, patch_size) if i in self.spatial_layer_indices else None
            self.blocks.append(PCFMBlock(dim, num_heads, max_seq_len=max_context_steps // patch_size,
                                         use_layer_scale=use_layer_scale, spatial_mixer=s_mixer,
                                         dedicated_spatial=enable_dedicated_spatial))
        self.norm_out = nn.RMSNorm(dim)

    def forward(self, data, training_mode='mae', mask_step=None, patch_mask=None, context_len=None):
        edge_index, edge_type, N_total = data.edge_index, getattr(data, 'edge_type', None), data.x.size(0)
        has_neighbors = torch.zeros((N_total, 1, 1), device=data.x.device, dtype=data.x.dtype)
        has_neighbors.index_fill_(0, edge_index[1][edge_index[0] != edge_index[1]], 1.0)
        z0 = self.embed(data.x, data, mask_step=mask_step, context_len=context_len, training_mode=training_mode)
        z = z0
        for block in self.blocks:
            z = block(z, edge_index, data.edge_attr, patch_mask=patch_mask, batch=data.batch,
                      num_graphs=data.num_graphs, x0=z0, has_neighbors=has_neighbors, edge_type=edge_type)
        return self.norm_out(z).to(z.dtype)


class MonotonicQuantileHead(nn.Module):
    def __init__(self, dim, patch_size, quantiles, dropout=0.0):
        super().__init__()
        self.patch_size, self.quantiles, self.num_quantiles = patch_size, quantiles, len(quantiles)
        self.median_idx = int(torch.argmin(torch.abs(torch.tensor(quantiles, dtype=torch.float32) - 0.5)).item())
        self.estimator = SwiGLUResidualBlock(dim, 256 * (((int(2 * dim * 4 / 3)) + 256 - 1) // 256),
                                             patch_size * self.num_quantiles, dropout)

    def forward(self, x):
        B, S, _ = x.shape
        out_reshaped = self.estimator(x).view(B, S, self.patch_size, self.num_quantiles)
        median = out_reshaped[..., self.median_idx:self.median_idx + 1]
        upper = (median + torch.cumsum(F.softplus(out_reshaped[..., self.median_idx + 1:]), dim=-1)
                 if self.median_idx + 1 < self.num_quantiles else out_reshaped[..., :0])
        lower = (torch.flip(median - torch.cumsum(torch.flip(F.softplus(out_reshaped[..., :self.median_idx]),
                                                             dims=[-1]), dim=-1), dims=[-1])
                 if self.median_idx > 0 else out_reshaped[..., :0])
        return torch.cat([lower, median, upper], dim=-1).view(B, S, -1)


# =========================================================================
# TASK WRAPPER (forecast + masked what-if inference)
# =========================================================================
class PCFMMAEWrapper(nn.Module):
    def __init__(self, backbone, dim, patch_size, pred_len, dropout: float = 0.0,
                 max_unrolled_chunks: int = 1, enable_dense_summary: bool = True,
                 enable_runtime_discrete_revin: bool = False, discrete_max_states: int = 20):
        super().__init__()
        self.backbone, self.dim, self.patch_size, self.pred_len = backbone, dim, patch_size, pred_len
        self.max_unrolled_chunks, self.enable_dense_summary = max_unrolled_chunks, enable_dense_summary
        self.quantiles = list(CONFIG["quantiles"])
        self.num_quantiles = len(self.quantiles)
        self.revin = PatchRobustRevIN(num_features=1,
                                      detect_discrete_runtime=enable_runtime_discrete_revin,
                                      discrete_max_states=discrete_max_states)
        self.head = MonotonicQuantileHead(dim, patch_size, self.quantiles, dropout=dropout)

    def forward(self, data, mode: str = 'mae', pred_len: int = None, mae_mask: torch.Tensor = None,
                denorm_mae: bool = False):
        dtype = next(self.backbone.parameters()).dtype
        if mode != 'forecast' and mae_mask is not None:
            data.m_pred = mae_mask.to(dtype)
        x, device = data.x.to(dtype), data.x.device
        pred_len = pred_len or self.pred_len
        max_steps, P = self.backbone.max_context_steps, self.patch_size
        if x.dim() == 2:
            x = x.unsqueeze(1)
        N, C, T_raw = x.shape

        pad_len_inf = P - (T_raw % P) if T_raw % P != 0 else 0
        if pad_len_inf > 0:
            x = F.pad(x, (pad_len_inf, 0), mode='constant', value=0.0)
        T_in = x.shape[-1]

        if mode == 'forecast':
            x_in, mask_step, m_pad, context_len, stats, P_hist = self._prepare_forecast_inputs(
                data, x, T_in, pred_len, pad_len_inf, P, max_steps, dtype, device)
        else:
            x_in, mask_step, m_pad, context_len, stats = self._prepare_mae_inputs(data, x, T_in, P, dtype, device)

        (x_in_bb, mask_bb, attn_bb, un_ei, un_ea, un_et, batch_vec, num_graphs, ctx_len,
         ptr, pad_len, K, x_trunc) = self._unroll_and_chunk_graph(
            data, x_in, mask_step, m_pad, context_len, mode, N, P, max_steps, dtype, device)
        bb_data = GraphInput(x=x_in_bb, edge_index=un_ei, edge_attr=un_ea, batch=batch_vec,
                             num_graphs=num_graphs, edge_type=un_et, ptr=ptr)
        latent = self.backbone(bb_data, training_mode=mode, mask_step=mask_bb,
                               patch_mask=attn_bb, context_len=ctx_len)
        out_flat = self.head(latent)
        full_pred = self._stitch_timeline(out_flat, K, N, latent.shape[1], P, pad_len)
        if mode == 'forecast':
            return self._decode_forecast(full_pred, stats, P_hist, P, pred_len)
        return self._decode_mae(full_pred, x_trunc, data, P, dtype, stats=stats, denorm=denorm_mae)

    def _prepare_forecast_inputs(self, data, x, T_in, pred_len, pad_len_inf, P, max_steps, dtype, device):
        N, C = x.shape[0], x.shape[1]
        is_discrete = getattr(data, 'is_discrete', None)
        max_hist = ((max_steps * self.max_unrolled_chunks - pred_len) // P) * P
        if T_in > max_hist:
            x = x[:, :, -max_hist:]
            T_in = max_hist
        P_hist, P_pred = T_in // P, math.ceil(pred_len / P)
        total_steps = (P_hist + P_pred) * P

        m_pad_full = torch.zeros((N, P_hist + P_pred, 1), device=device, dtype=dtype)
        m_pad_step_full = torch.zeros((N, total_steps, 1), device=device, dtype=dtype)
        if pad_len_inf > 0:
            m_pad_step_full[:, :pad_len_inf, :] = 1.0
            if (pad_len_inf // P) > 0:
                m_pad_full[:, :(pad_len_inf // P), :] = 1.0

        m_pred_full = torch.zeros((N, P_hist + P_pred, 1), device=device, dtype=dtype)
        m_pred_full[:, P_hist:, :] = 1.0

        step_mask_in = m_pad_step_full[:, :T_in, :]
        x_norm, stats = self.revin(x, 'norm', mask=step_mask_in.transpose(1, 2).contiguous(),
                                   is_discrete=is_discrete)
        x_in = torch.zeros((N, C, total_steps), device=device, dtype=dtype)
        x_in[:, :, :T_in] = x_norm

        mask_step = torch.max(m_pad_full, m_pred_full).repeat_interleave(P, dim=1)
        mask_step[:, :T_in, :] = torch.max(mask_step[:, :T_in, :], m_pad_step_full[:, :T_in, :])
        return x_in, mask_step.transpose(1, 2).contiguous(), m_pad_full, P_hist * P, stats, P_hist

    def _prepare_mae_inputs(self, data, x, T_in, P, dtype, device):
        N = x.shape[0]
        is_discrete = getattr(data, 'is_discrete', None)
        m_pad = data.m_pad.to(dtype) if hasattr(data, 'm_pad') else torch.zeros((N, T_in // P, 1), device=device, dtype=dtype)
        m_pred = data.m_pred.to(dtype) if hasattr(data, 'm_pred') else torch.zeros((N, T_in // P, 1), device=device, dtype=dtype)
        m_pad_step = m_pad.repeat_interleave(P, dim=1)
        m_pred_step = m_pred.repeat_interleave(P, dim=1)
        mask_step = torch.max(m_pad_step, m_pred_step)
        x_norm, stats = self.revin(x, 'norm', mask=mask_step.transpose(1, 2).contiguous(),
                                   is_discrete=is_discrete)
        return x_norm, mask_step.transpose(1, 2).contiguous(), m_pad, x_norm.shape[2], stats

    def _unroll_and_chunk_graph(self, data, x_in, mask_step_x, attn_mask, context_len, mode, N, P,
                                max_steps, dtype, device):
        T_actual, x_trunc = x_in.shape[-1], x_in
        if T_actual > max_steps:
            K = math.ceil(T_actual / max_steps)
            if K > self.max_unrolled_chunks:
                allowed_T = self.max_unrolled_chunks * max_steps
                x_in, mask_step_x = x_in[:, :, -allowed_T:], mask_step_x[:, :, -allowed_T:]
                if attn_mask is not None:
                    attn_mask = attn_mask[:, -(allowed_T // P):, :]
                T_actual, K, x_trunc = allowed_T, self.max_unrolled_chunks, x_in
            pad_len = (K * max_steps) - T_actual
            if pad_len > 0:
                x_in = F.pad(x_in, (pad_len, 0), value=0.0)
                mask_step_x = F.pad(mask_step_x, (pad_len, 0), value=1.0)
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask.transpose(1, 2), (pad_len // P, 0), value=1.0).transpose(1, 2)
            x_in_bb = x_in.view(N, 1, K, max_steps).permute(2, 0, 1, 3).reshape(N * K, 1, max_steps)
            mask_bb = mask_step_x.view(N, 1, K, max_steps).permute(2, 0, 1, 3).reshape(N * K, 1, max_steps)
            attn_bb = (attn_mask.view(N, K, max_steps // P, 1).permute(1, 0, 2, 3).reshape(N * K, max_steps // P, 1)
                       if attn_mask is not None else None)
            ei_list, ea_list, et_list = [], [], []
            base_ea = (data.edge_attr.to(dtype) if getattr(data, 'edge_attr', None) is not None
                       else torch.zeros((data.edge_index.size(1), 1), device=device, dtype=dtype))
            for k in range(K):
                ei_list.append(data.edge_index + (k * N))
                ea_list.append(base_ea)
                et_list.append(torch.zeros(data.edge_index.size(1), dtype=torch.long, device=device))
            for k in range(1, K):
                for prev_k in range(0, k if self.enable_dense_summary else k - 1, 1):
                    ei_list.append(torch.stack([torch.arange(N, device=device) + (prev_k * N),
                                                torch.arange(N, device=device) + (k * N)], dim=0))
                    ea_list.append(torch.zeros((N, 1), device=device, dtype=dtype))
                    et_list.append(torch.ones(N, dtype=torch.long, device=device))
            un_ei = torch.cat(ei_list, dim=1)
            un_ea = torch.cat(ea_list, dim=0)
            un_et = torch.cat(et_list, dim=0)
            batch_vec, num_graphs, ctx_len, ptr = data.batch.repeat(K), data.num_graphs * K, max_steps, None
        else:
            K, pad_len = 1, 0
            x_in_bb, mask_bb, attn_bb = x_in, mask_step_x, attn_mask
            un_ei = data.edge_index
            un_ea = data.edge_attr.to(dtype) if getattr(data, 'edge_attr', None) is not None else None
            un_et = torch.zeros(un_ei.size(1), dtype=torch.long, device=device) if un_ei.size(1) > 0 else None
            batch_vec, num_graphs, ctx_len, ptr = data.batch, data.num_graphs, context_len, getattr(data, 'ptr', None)
        return (x_in_bb, mask_bb, attn_bb, un_ei, un_ea, un_et, batch_vec, num_graphs, ctx_len,
                ptr, pad_len, K, x_trunc)

    def _stitch_timeline(self, out_flat, K, N, P_actual, P, pad_len):
        if K > 1:
            out_reshaped = out_flat.reshape(K, N, P_actual, P, self.num_quantiles)
            full_pred = out_reshaped.permute(1, 0, 2, 3, 4).contiguous().reshape(N, -1, self.num_quantiles)
            return full_pred[:, pad_len:, :]
        return out_flat.reshape(N, P_actual, P, self.num_quantiles).reshape(N, -1, self.num_quantiles)

    def _decode_forecast(self, full_pred, stats, P_hist, P, pred_len):
        pred_fut = full_pred[:, P_hist * P: P_hist * P + pred_len, :]
        mean, scale, s_min, s_max, route_mask, is_dynamic = stats
        last_stats = (mean[..., -1:], scale[..., -1:], s_min[..., -1:], s_max[..., -1:],
                      route_mask[..., -1:], is_dynamic[..., -1:])
        pred_denorm = self.revin(pred_fut.permute(0, 2, 1).contiguous(), 'denorm', stats=last_stats)
        return pred_denorm.permute(0, 2, 1).contiguous()

    def _decode_mae(self, full_pred, x_trunc, data, P, dtype, stats=None, denorm=False):
        if denorm and stats is not None:
            full_pred = self.revin(full_pred.permute(0, 2, 1).contiguous(), 'denorm',
                                   stats=stats).permute(0, 2, 1).contiguous()
        return full_pred


# =========================================================================
# GRAPH ESTIMATION ("CP" mode -- Granger + cross-correlation, needs scipy)
# =========================================================================
def _z(a: np.ndarray, eps=1e-9) -> np.ndarray:
    return (a - a.mean()) / (a.std() + eps)


def _block_average(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    T = (arr.shape[-1] // factor) * factor
    return arr[..., :T].reshape(*arr.shape[:-1], -1, factor).mean(axis=-1)


def _ridge_lstsq(Z: np.ndarray, Y: np.ndarray, alpha=1e-3):
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ Y)


class RobustHybridGrangerCCEstimator:
    """Pairwise Granger-causality gate + cross-correlation lag assignment
    (verbatim from the PCFM evaluation code; validated defaults of the best
    GiftEval runs: p=0.05, 256-lag budget, downsample factors (16, 8),
    edge_lag_type='cc')."""

    def __init__(self, max_lag=256, p_value_threshold=0.05, edge_lag_type="cc",
                 downsample_factors=(16, 8), ridge_alpha=1e-3, eps=1e-9):
        self.max_lag, self.p_value_threshold = max_lag, p_value_threshold
        self.edge_lag_type, self.downsample_factors = edge_lag_type, downsample_factors
        self.ridge_alpha, self.eps = ridge_alpha, eps

    def _is_non_stationary(self, x: np.ndarray, threshold=0.98) -> bool:
        if x.std() < self.eps:
            return False
        rho = np.corrcoef(x[1:], x[:-1])[0, 1]
        return False if np.isnan(rho) else abs(rho) >= threshold

    def prepare(self, X: np.ndarray):
        N, T = X.shape
        X_prep, is_stationary, is_degenerate = [], [], []
        for i in range(N):
            series = X[i]
            if series.std() < self.eps:
                X_prep.append(series); is_stationary.append(True); is_degenerate.append(True)
                continue
            is_degenerate.append(False)
            series_std = _z(series, self.eps)
            if not self._is_non_stationary(series_std):
                X_prep.append(series_std); is_stationary.append(True)
            else:
                X_prep.append(_z(np.diff(series_std), self.eps)); is_stationary.append(False)
        X_aligned = np.array([X_prep[i][1:] if is_stationary[i] else X_prep[i] for i in range(N)])
        return (X_aligned, is_degenerate), T - 1

    def estimate_pair(self, prep_data, s_idx: int, d_idx: int):
        from scipy.stats import f as f_dist
        X_aligned, is_degenerate = prep_data
        if is_degenerate[s_idx] or is_degenerate[d_idx]:
            return 0.0, 0.0
        x_raw, y_raw = X_aligned[s_idx], X_aligned[d_idx]
        best = None   # (p_val, scale, coef_U, x_s, y_s, T_s, p_used)
        for S in self.downsample_factors:
            x_s, y_s = _block_average(x_raw, S), _block_average(y_raw, S)
            T_s = x_s.shape[-1]
            max_feasible = (T_s - 5) // 3
            if max_feasible < 1:
                continue
            p_local = min(self.max_lag, max_feasible)
            Y_pred = y_s[p_local:]
            Z_R_cols = [np.ones(T_s - p_local)]
            for i in range(1, p_local + 1):
                Z_R_cols.append(y_s[p_local - i: T_s - i])
            Z_R = np.column_stack(Z_R_cols)
            Z_U = np.column_stack(Z_R_cols + [x_s[p_local - i: T_s - i] for i in range(1, p_local + 1)])
            try:
                coef_R = _ridge_lstsq(Z_R, Y_pred, self.ridge_alpha)
                coef_U = _ridge_lstsq(Z_U, Y_pred, self.ridge_alpha)
                ssr_R = np.sum((Y_pred - Z_R @ coef_R) ** 2)
                ssr_U = np.sum((Y_pred - Z_U @ coef_U) ** 2)
                if ssr_U < self.eps:
                    continue
                df_num, df_den = p_local, T_s - 3 * p_local - 1
                F_stat = ((ssr_R - ssr_U) / df_num) / (ssr_U / df_den)
                p_val = float(f_dist.sf(F_stat, df_num, df_den))
                if best is None or p_val < best[0]:
                    best = (p_val, S, coef_U, x_s, y_s, T_s, p_local)
            except Exception:
                continue
        if best is None:
            return 0.0, 0.0
        p_val, scale, coef_U, x_s, y_s, T_s, p_used = best
        if p_val >= self.p_value_threshold * np.sqrt(100 / max(100, T_s)):
            return 0.0, 0.0
        strength = float(-np.log10(p_val + self.eps))
        if self.edge_lag_type == "granger":
            lag = float(np.argmax(np.abs(coef_U[p_used + 1:])) + 1) * scale
        else:                                                     # 'cc' (validated default)
            n_fft = 2 * T_s - 1
            corr_lin = np.fft.irfft(np.fft.rfft(y_s, n=n_fft) * np.conj(np.fft.rfft(x_s, n=n_fft)), n=n_fft)
            corr_ctr = np.concatenate([corr_lin[-(T_s - 1):], corr_lin[:T_s]], axis=-1)
            lag = float(np.argmax(np.abs(corr_ctr[T_s: T_s + p_used])) + 1) * scale
        return lag, strength


# =========================================================================
# PUBLIC API
# =========================================================================
GRAPH_MODES = {
    "CI": "channel-independent: self-loops only, no cross-node interaction",
    "CD": "channel-dependent: fully connected with zero lags (default)",
    "CP": "channel-partial: estimated sparse graph with lags (Granger+CC, needs scipy)",
}


class PCFMPredictor:
    """User-facing interface around the pretrained PCFM checkpoint.

    Tensor API -- one multivariate system as [num_nodes, time] (float, no NaNs):

        forecast(history, prediction_length, graph="CD", ...)
        what_if(window, context_length, masked_nodes, graph="CD", ...)
        estimate_graph(history)  ->  (edge_index, edge_lags)

    DataFrame API (long format, Chronos-style):

        predict_df(context_df, future_df=None, prediction_length=64,
                   quantile_levels=(0.1, 0.5, 0.9), id_column="id",
                   timestamp_column="timestamp", target="target", graph="CD")

    The `graph` argument is the same everywhere and accepts
        "CI" | "CD" | "CP"                       (see GRAPH_MODES)
        [2, E] edge index (+ optional edge_lags in time steps)
        [(src, dst), ...] or [(src, dst, lag), ...] tuples -- node indices in
        the tensor API, COLUMN NAMES in the DataFrame API.
    Self-loops are added automatically; lags are always optional (0 = the
    source's previous patch is tapped).
    """

    def __init__(self, wrapper: PCFMMAEWrapper, device: str):
        self.model = wrapper
        self.device = device
        self.patch_size = wrapper.patch_size
        self.quantile_levels = list(wrapper.quantiles)
        self.median_idx = wrapper.head.median_idx

    # ------------------------------------------------------------------ load
    @classmethod
    def from_pretrained(cls, repo_or_path: str = HF_REPO_ID, filename: Optional[str] = None,
                        device: Optional[str] = None,
                        runtime_discrete_revin: Optional[bool] = None) -> "PCFMPredictor":
        """Load the checkpoint from a local .pt file / directory or from the
        Hugging Face Hub. `runtime_discrete_revin` overrides the discrete-RevIN
        routing (None = keep the checkpoint's configuration)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = cls._resolve_checkpoint(repo_or_path, filename)

        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict, saved_cfg = load_file(path), {}
        else:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                saved_cfg = ckpt.get("config", {}) or {}
            else:                               # raw state dict
                state_dict, saved_cfg = ckpt, {}

        cfg = dict(CONFIG)
        for key in ("dim", "patch_size", "num_heads", "max_context_steps", "pred_len",
                    "use_layer_scale", "enable_dedicated_spatial", "max_unrolled_chunks",
                    "enable_dense_summary", "enable_runtime_discrete_revin", "discrete_max_states"):
            if key in saved_cfg:
                cfg[key] = saved_cfg[key]
        cfg["layers"] = saved_cfg.get("layers", saved_cfg.get("num_layers", cfg["layers"]))

        backbone = PCFM(dim=cfg["dim"], patch_size=cfg["patch_size"], layers=cfg["layers"],
                        num_heads=cfg["num_heads"], max_context_steps=cfg["max_context_steps"],
                        use_layer_scale=cfg["use_layer_scale"],
                        enable_dedicated_spatial=cfg["enable_dedicated_spatial"],
                        spatial_stride=cfg["spatial_stride"])
        wrapper = PCFMMAEWrapper(backbone, cfg["dim"], cfg["patch_size"], cfg["pred_len"],
                                 max_unrolled_chunks=cfg["max_unrolled_chunks"],
                                 enable_dense_summary=cfg["enable_dense_summary"],
                                 enable_runtime_discrete_revin=cfg["enable_runtime_discrete_revin"],
                                 discrete_max_states=cfg["discrete_max_states"])
        clean_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        wrapper.load_state_dict(clean_sd, strict=True)
        if runtime_discrete_revin is not None:
            wrapper.revin.detect_discrete_runtime = bool(runtime_discrete_revin)
        wrapper.to(device).eval()
        n_params = sum(p.numel() for p in wrapper.parameters())
        print(f"[PCFM] loaded {path} ({n_params/1e6:.1f}M params) on {device} | "
              f"discrete-RevIN routing: {'ON' if wrapper.revin.detect_discrete_runtime else 'OFF'}")
        return cls(wrapper, device)

    @staticmethod
    def _resolve_checkpoint(repo_or_path: str, filename: Optional[str]) -> str:
        if os.path.isfile(repo_or_path):
            return repo_or_path
        exts = (".safetensors", ".pt", ".pth", ".bin")
        if os.path.isdir(repo_or_path):
            cands = sorted((f for f in os.listdir(repo_or_path) if f.endswith(exts)),
                           key=lambda f: exts.index("." + f.rsplit(".", 1)[-1]))
            if not cands:
                raise FileNotFoundError(f"No checkpoint ({'/'.join(exts)}) in directory: {repo_or_path}")
            return os.path.join(repo_or_path, filename or cands[0])
        # otherwise: Hugging Face Hub repo id
        from huggingface_hub import hf_hub_download, list_repo_files
        if filename is None:
            files = list_repo_files(repo_or_path)
            cands = sorted((f for f in files if f.endswith(exts)),
                           key=lambda f: exts.index("." + f.rsplit(".", 1)[-1]))
            if not cands:
                raise FileNotFoundError(f"No checkpoint ({'/'.join(exts)}) found in Hub repo {repo_or_path}: {files}")
            filename = cands[0]
        return hf_hub_download(repo_id=repo_or_path, filename=filename)

    # ------------------------------------------------------------- utilities
    def _to_tensor(self, x) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2:
            raise ValueError(f"expected [num_nodes, time] (or [time]), got shape {tuple(t.shape)}")
        if torch.isnan(t).any():
            raise ValueError("input contains NaNs -- impute missing values first")
        return t.to(self.device)

    @torch.no_grad()
    def estimate_graph(self, history, top_k: int = 10, p_value: float = 0.05,
                       max_lag_steps: int = 256,
                       names: Optional[Sequence[str]] = None):
        """Estimate a sparse causal graph WITH transport lags from data (the
        "CP" mode): every ordered node pair is Granger-tested (F-test on ridge
        VAR residuals over block-averaged copies), significant sources are kept
        up to `top_k` per destination, and each edge's lag comes from the
        cross-correlation peak -- the validated configuration of the PCFM
        evaluation. Candidate edges whose estimated lag exceeds `max_lag_steps`
        (default 256 = the evaluation's 8-patch lag budget) are discarded:
        single-window cross-correlation peaks at extreme lags are overwhelmingly
        spurious. Returns (edge_index [2, E], edge_lags [E]) incl. self-loops.
        Requires scipy."""
        x = self._to_tensor(history).cpu().numpy()
        N = x.shape[0]
        est = RobustHybridGrangerCCEstimator(p_value_threshold=p_value)
        prep, _ = est.prepare(x)
        edges = [[i, i] for i in range(N)]
        lags = [0.0] * N
        for d in range(N):
            cands = []
            for s in range(N):
                if s == d:
                    continue
                lag, strength = est.estimate_pair(prep, s, d)
                if strength > 0.0 and 0.0 < lag <= max_lag_steps:
                    cands.append((strength, s, lag))
            for _, s, lag in sorted(cands, reverse=True)[:top_k]:
                edges.append([s, d])
                lags.append(float(lag))
        ei = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
        ea = torch.tensor(lags, dtype=torch.float32, device=self.device)
        return ei, ea

    def _resolve_graph(self, graph, edge_lags, num_nodes: int, x_context: torch.Tensor,
                       names: Optional[Sequence[str]] = None):
        """Turn the `graph` argument into (edge_index [2,E], edge_attr [E,1]),
        always including self-loops."""
        if graph is None:
            graph = "CD"
        if isinstance(graph, str):
            g = graph.strip().upper()
            if g in ("CD", "DENSE", "FULLY_CONNECTED", "FULL"):
                ei = fully_connected_graph(num_nodes, device=self.device)
                ea = torch.zeros(ei.size(1), device=self.device)
            elif g in ("CI", "INDEPENDENT", "SELF", "UNIVARIATE"):
                idx = torch.arange(num_nodes, device=self.device)
                ei = torch.stack([idx, idx], dim=0)
                ea = torch.zeros(num_nodes, device=self.device)
            elif g in ("CP", "ESTIMATED", "SPARSE", "AUTO"):
                ei, ea = self.estimate_graph(x_context)
            else:
                raise ValueError(f"unknown graph mode '{graph}' -- use " + " | ".join(GRAPH_MODES))
        elif isinstance(graph, (list, tuple)) and len(graph) > 0 \
                and isinstance(graph[0], (list, tuple)) and len(graph[0]) in (2, 3) \
                and not (len(graph) == 2 and np.isscalar(graph[0][0]) and len(graph[0]) != 2):
            # [(src, dst), ...] or [(src, dst, lag), ...]; str entries resolved via `names`
            name_to_idx = {str(n): i for i, n in enumerate(names)} if names is not None else {}

            def _idx(v):
                if isinstance(v, str):
                    if v not in name_to_idx:
                        raise ValueError(f"unknown node/column name '{v}' (known: {list(name_to_idx)})")
                    return name_to_idx[v]
                return int(v)
            src, dst, lg = [], [], []
            for e in graph:
                src.append(_idx(e[0])); dst.append(_idx(e[1]))
                lg.append(float(e[2]) if len(e) == 3 else 0.0)
            # add self-loops for every node
            for i in range(num_nodes):
                src.append(i); dst.append(i); lg.append(0.0)
            ei = torch.tensor([src, dst], dtype=torch.long, device=self.device)
            ea = torch.tensor(lg, dtype=torch.float32, device=self.device)
        else:                                              # [2, E] array / tensor
            ei = torch.as_tensor(graph, dtype=torch.long, device=self.device)
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError(f"edge index must be [2, E], got {tuple(ei.shape)}")
            if edge_lags is None:
                ea = torch.zeros(ei.size(1), device=self.device)
            else:
                ea = torch.as_tensor(edge_lags, dtype=torch.float32, device=self.device).view(-1)
                if ea.numel() != ei.size(1):
                    raise ValueError(f"edge_lags has {ea.numel()} entries for {ei.size(1)} edges")
            missing = [i for i in range(num_nodes)
                       if not bool(((ei[0] == i) & (ei[1] == i)).any())]
            if missing:
                loops = torch.tensor([missing, missing], dtype=torch.long, device=self.device)
                ei = torch.cat([ei, loops], dim=1)
                ea = torch.cat([ea, torch.zeros(len(missing), device=self.device)])
        if ei.numel() and int(ei.max()) >= num_nodes:
            raise ValueError("graph references a node index >= num_nodes")
        return ei, ea.view(-1, 1)

    def _make_data(self, x: torch.Tensor, graph, edge_lags, x_context=None, names=None) -> GraphInput:
        N = x.size(0)
        ei, ea = self._resolve_graph(graph, edge_lags, N,
                                     x if x_context is None else x_context, names=names)
        return GraphInput(x=x, edge_index=ei, edge_attr=ea,
                          batch=torch.zeros(N, dtype=torch.long, device=self.device),
                          num_graphs=1)

    def _interp_quantiles(self, q: torch.Tensor, levels: Sequence[float]) -> np.ndarray:
        """Linear interpolation of arbitrary quantile levels from the model's 9
        native levels (0.1..0.9); levels outside are clamped to the edges."""
        native = np.asarray(self.quantile_levels)
        req = np.clip(np.asarray(list(levels), dtype=float), native[0], native[-1])
        idx = np.clip(np.searchsorted(native, req, side="left"), 1, len(native) - 1)
        lo, hi = native[idx - 1], native[idx]
        w = (req - lo) / (hi - lo)
        q_np = q.detach().cpu().numpy()
        return q_np[..., idx - 1] * (1 - w) + q_np[..., idx] * w

    # -------------------------------------------------------------- forecast
    @torch.no_grad()
    def forecast(self, history, prediction_length: int = None, graph="CD", edge_lags=None,
                 return_quantiles: bool = False):
        """Forecast every node `prediction_length` steps beyond the given
        [num_nodes, T] history. Returns the median forecast
        [num_nodes, prediction_length], or all 9 native quantiles
        [num_nodes, prediction_length, 9] with return_quantiles=True."""
        x = self._to_tensor(history)
        prediction_length = prediction_length or self.model.pred_len
        data = self._make_data(x, graph, edge_lags)
        q = self.model(data, mode='forecast', pred_len=prediction_length)   # [N, F, 9]
        return q if return_quantiles else q[..., self.median_idx]

    # --------------------------------------------------------------- what-if
    @torch.no_grad()
    def what_if(self, window, context_length: int = None, masked_nodes: Sequence[int] = None,
                graph="CD", edge_lags=None, custom_mask=None, glue: bool = True,
                return_quantiles: bool = False):
        """Masked what-if / counterfactual inference.

        `window` [num_nodes, T] contains history AND future for every node.
        Beyond `context_length`, the futures of `masked_nodes` are hidden and
        re-predicted, while every OTHER node's future stays observed and acts
        as the intervention forcing (e.g. an actuator plan you wrote into the
        window). Arbitrary lengths are handled by internal patch-aligned
        padding (edge replication, sliced away before returning); when "CP" is
        requested the graph is estimated from the CONTEXT only (no leakage
        from the intervention plan).

        `custom_mask` replaces the default mask entirely: [num_nodes, P] or
        [num_nodes, T] with 1 = hide & predict, 0 = observed (step-level masks
        are pooled to patches; T must then be a multiple of patch_size=32).

        Returns the full timeline [num_nodes, T] in raw units (or
        [num_nodes, T, 9] quantiles): observed regions are copied from the
        input, masked regions are model predictions. With glue=True (and a
        default mask) each masked node's future is shifted additively so it
        continues from its last observed value."""
        x = self._to_tensor(window)
        N, T = x.shape
        P = self.patch_size

        if custom_mask is not None:
            if T % P != 0:
                raise ValueError(f"custom_mask requires a window length that is a multiple of {P}")
            P_total = T // P
            m = torch.as_tensor(custom_mask, dtype=torch.float32, device=self.device)
            if m.shape == (N, T):                      # step-level -> patch-level (any-step rule)
                m = m.view(N, P_total, P).amax(dim=-1)
            if m.shape != (N, P_total):
                raise ValueError(f"custom_mask must be [N, {P_total}] (patches) or [N, {T}] (steps)")
            mae_mask = (m > 0.5).float().unsqueeze(-1)
            x_pad, L0, T_pad, ctx_pad = x, 0, T, None
            glue = False
        else:
            if context_length is None or masked_nodes is None:
                raise ValueError("provide context_length and masked_nodes, or a custom_mask")
            if not (0 < context_length < T):
                raise ValueError(f"context_length must lie strictly inside the window (0, {T})")
            # patch-align by edge replication: left-pad the context, right-pad the future
            L0 = (-context_length) % P
            F_pad = (-(T - context_length)) % P
            x_pad = torch.cat([x[:, :1].expand(N, L0), x, x[:, -1:].expand(N, F_pad)], dim=1)
            T_pad, ctx_pad = T + L0 + F_pad, context_length + L0
            mae_mask = torch.zeros((N, T_pad // P, 1), device=self.device)
            for n in masked_nodes:
                mae_mask[int(n), ctx_pad // P:, 0] = 1.0

        data = self._make_data(x_pad, graph, edge_lags,
                               x_context=(x[:, :context_length] if custom_mask is None else x))
        recon_q = self.model(data, mode='mae', mae_mask=mae_mask, denorm_mae=True)  # [N, T_pad, 9]

        # observed regions: keep the input values; masked regions: model prediction
        step_mask = mae_mask.repeat_interleave(P, dim=1)                            # [N, T_pad, 1]
        out_q = torch.where(step_mask.bool(), recon_q, x_pad.unsqueeze(-1).expand_as(recon_q))
        out_q = out_q[:, L0:L0 + T]                                                 # drop padding

        if glue and custom_mask is None:
            for n in masked_nodes:
                n = int(n)
                offset = x[n, context_length - 1] - out_q[n, context_length, self.median_idx]
                out_q[n, context_length:, :] += offset
        return out_q if return_quantiles else out_q[..., self.median_idx]

    # ------------------------------------------------------- DataFrame API
    @torch.no_grad()
    def predict_df(self, context_df, future_df=None, prediction_length: int = 64,
                   quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
                   id_column: str = "id", timestamp_column: str = "timestamp",
                   target: Union[str, Sequence[str]] = "target",
                   graph="CD", edge_lags=None):
        """Chronos-style long-format pandas interface.

        `context_df` holds the past of every column; every numeric column that
        is not a target (and not id/timestamp) is treated as a covariate.
        `future_df` (optional) holds FUTURE values of covariates -- e.g. known
        schedules, or an intervention plan for a what-if question. Covariates
        present in `future_df` stay observed over the horizon; targets (and
        covariates absent from `future_df`) are predicted.

        graph: "CI" | "CD" | "CP", a [2, E] index over the node order
        [targets..., covariates...] (df column order), or name tuples like
        [("valve", "temp", 64.0)].

        Returns a long DataFrame with columns
            [id_column?, timestamp_column, "target_column", "predictions",
             str(q) for q in quantile_levels]
        where "predictions" is the median forecast in raw units.
        """
        import pandas as pd

        targets = [target] if isinstance(target, str) else list(target)
        has_id = id_column in context_df.columns
        has_ts = timestamp_column in context_df.columns

        value_cols = [c for c in context_df.columns
                      if c not in (id_column, timestamp_column)
                      and np.issubdtype(np.asarray(context_df[c]).dtype, np.number)]
        for t in targets:
            if t not in value_cols:
                raise ValueError(f"target column '{t}' not found among numeric columns {value_cols}")
        covariates = [c for c in value_cols if c not in targets]
        columns = targets + covariates                       # node order
        known_covs = ([c for c in covariates if future_df is not None and c in future_df.columns])

        # resolve column-name graph tuples to node indices once
        if isinstance(graph, (list, tuple)) and len(graph) > 0 and isinstance(graph[0], (list, tuple)):
            col_idx = {c: i for i, c in enumerate(columns)}

            def _cidx(v):
                if isinstance(v, str):
                    if v not in col_idx:
                        raise ValueError(f"graph references unknown column '{v}' (columns: {columns})")
                    return col_idx[v]
                return int(v)
            graph = [((_cidx(e[0]), _cidx(e[1]), float(e[2])) if len(e) == 3
                      else (_cidx(e[0]), _cidx(e[1]))) for e in graph]

        groups = (list(context_df.groupby(id_column, sort=False)) if has_id
                  else [(None, context_df)])
        fut_groups = {}
        if future_df is not None:
            fut_groups = ({k: g for k, g in future_df.groupby(id_column, sort=False)}
                          if has_id and id_column in future_df.columns
                          else {None: future_df})

        rows = []
        for gid, g in groups:
            g = g.sort_values(timestamp_column) if has_ts else g
            hist = np.asarray(g[columns], dtype=np.float32).T          # [N, T_ctx]
            T_ctx = hist.shape[1]
            gf = fut_groups.get(gid) if future_df is not None else None
            if gf is not None:
                gf = gf.sort_values(timestamp_column) if (has_ts and timestamp_column in gf.columns) else gf
                if len(gf) < prediction_length:
                    raise ValueError(f"future_df must cover prediction_length={prediction_length} "
                                     f"rows for id={gid!r} (got {len(gf)})")
                gf = gf.iloc[:prediction_length]

            if gf is None:
                q9 = self.forecast(hist, prediction_length, graph=graph,
                                   edge_lags=edge_lags, return_quantiles=True)   # [N, F, 9]
                pred_q9 = {t: q9[columns.index(t)] for t in targets}
            else:
                window = np.concatenate(
                    [hist, np.repeat(hist[:, -1:], prediction_length, axis=1)], axis=1)
                for c in known_covs:
                    window[columns.index(c), T_ctx:] = np.asarray(gf[c], dtype=np.float32)
                masked = [columns.index(t) for t in targets] + \
                         [columns.index(c) for c in covariates if c not in known_covs]
                recon = self.what_if(window, context_length=T_ctx, masked_nodes=masked,
                                     graph=graph, edge_lags=edge_lags,
                                     return_quantiles=True)                       # [N, T, 9]
                pred_q9 = {t: recon[columns.index(t), T_ctx:] for t in targets}

            # future timestamps: from future_df when available, else inferred
            if gf is not None and has_ts and timestamp_column in gf.columns:
                ts_fut = list(gf[timestamp_column])
            elif has_ts:
                ts = pd.to_datetime(g[timestamp_column])
                freq = pd.infer_freq(ts) if len(ts) >= 3 else None
                step = pd.tseries.frequencies.to_offset(freq) if freq else (ts.iloc[-1] - ts.iloc[-2])
                ts_fut = [ts.iloc[-1] + step * (i + 1) for i in range(prediction_length)]
            else:
                ts_fut = list(range(T_ctx, T_ctx + prediction_length))

            for t in targets:
                q_req = self._interp_quantiles(pred_q9[t], quantile_levels)       # [F, len(levels)]
                med = pred_q9[t][:, self.median_idx].cpu().numpy()
                block = {timestamp_column: ts_fut, "target_column": t, "predictions": med}
                if has_id:
                    block = {id_column: gid, **block}
                for j, lv in enumerate(quantile_levels):
                    block[str(lv)] = q_req[:, j]
                rows.append(pd.DataFrame(block))
        return pd.concat(rows, ignore_index=True)
