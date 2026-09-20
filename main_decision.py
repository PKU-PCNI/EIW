# -*- coding: utf-8 -*-
"""
RL for single-user MCS (modulation order) selection with sparse pilots in an OFDM/LTE-like system.

Updates include:
- Reward smoothing with sigmoid thresholds to avoid hard 0/1 rewards.
- Stronger exploration (epsilon-greedy, temperature scaling, and higher entropy regularization).
- Advantage normalization to stabilize REINFORCE.
- Optional SNR domain randomization during training.
- Logging of detailed rate statistics (sum-rate and per-TTI averages).

Replace H_train / H_val with your own channel tensor H (B_total, L, N) complex64 if desired.
"""
import math
import os
import random
from contextlib import suppress
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt       

try:  # Optional, only needed for .mat loading
    import scipy.io as sio
except ImportError:  # pragma: no cover - scipy optional
    sio = None

# -----------------------------       
# Hyperparameters (editable)
# -----------------------------
SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset / system
DATA_SOURCE = "mat"   # "synthetic" or "mat"
MAT_DATA_PATH = "./channel_gen/dataset/UMa_Re.mat"
MAT_VAR_NAME  = "H_cellfree_time"
MAT_VAL_RATIO = 0.2          # Fraction of episodes reserved for validation when using MAT source
BATCH_SIZE    = 800
L             = 50       # Trajectory length (TTIs)
N             = 128      # Number of subcarriers
GAIN_MIN = 0.5
GAIN_MAX = 10

PILOT_STRIDE  = 16        # Sparse pilot stride (N/4 pilots)
W             = 1.92e6   # Normalized bandwidth (Hz)
R_C           = 0.8      # Fixed code rate
P_DAT         = 1.0      # Data symbol power
P_PIL         = 1.0      # Pilot symbol power
SIGMA2_DB     = -116   # Noise power (dB)
SIGMA2_LIN    = 10.0 ** (SIGMA2_DB / 10.0)  # Linear domain noise power
RHO           = 0.9     # Gauss-Markov time correlation coefficient

# RL / training
EPOCHS        = 80
DISCOUNT      = 0
LR            = 3e-3     # 5 for prior-only, 2 for pilot-only
ENTROPY_COEF  = 0.01     # Stronger entropy bonus
GRAD_CLIP     = 1.0
TEMPERATURE   = 1.0      # Softmax temperature
EPSILON       = 0.05     # Epsilon-greedy probability

# Networks
D_U           = 64       # UE embedding dimension
D_PI          = 8        # Prior dimension (kept for API compatibility, zeroed here)
HIDDEN_UE     = 384
HIDDEN_BS     = 384
PRIOR_MODE    = "route"   # 'pilot_only', 'prior_only', 'both', 'route'
PRIOR_SNR_MIN_DB = -5.0
PRIOR_SNR_MAX_DB = 35.0
PRIOR_NOISE_STD_DB = 3.0  # dB RMS noise added to prior measurement

# Route-mode inference/budget regularization
ROUTE_MODE_A_COST = 0   # Expert A (prior-only) inference cost
ROUTE_MODE_B_COST = 1  # Expert B (prior+UE) inference cost
INFER_LAMBDA      = 0.00   # Weight for the inference-cost penalty
# 0.1 0.08 0.05 0.03 0.01

ROUTE_GUMBEL_TAU  = 0.8   # Temperature for straight-through Gumbel-softmax gating
EPOCH_PRE         = 50    # Uniform expert pre-training epochs
EPOCH_PRE_GATE    = 30    # Gate-only fine-tune epochs with infer_loss

# Action space (modulation order choices only)
MCS_LIST      = [2, 4, 16, 64, 256]   # BPSK, QPSK, 16QAM, 64QAM, 256QAM
MCS_THRESH_DB = {2: -3.0, 4: 2.0, 16: 8.0, 64: 14.0, 256: 20.0}  # Simplified thresholds (dB)

# Reward smoothing
USE_SMOOTH_REWARD = True
TAU_DB            = 2.0   # Sigmoid transition width (dB)

# Domain randomization: per-TTI SNR jitter during training (dB)
SNR_JITTER_DB     = 1   # Set to 0.0 to disable

# -----------------------------
# Utility functions
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def power_to_db(power: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Convert power (linear) tensor to dB, clamped for stability."""
    return 10.0 * torch.log10(torch.clamp(power, min=eps))

def complex_normal(shape, dtype=torch.complex64, device="cpu", scale=1.0):
    real = torch.randn(shape, device=device) * (scale / math.sqrt(2))
    imag = torch.randn(shape, device=device) * (scale / math.sqrt(2))
    return torch.complex(real, imag).to(dtype)

def get_pilot_indices(N: int, stride: int):
    return torch.arange(0, N, stride, dtype=torch.long)

def ls_estimate_from_pilots(H_t: torch.Tensor,
                            pilot_idx: torch.Tensor,
                            P_pil: float,                # 保留参数以兼容旧API，不再使用
                            sigma2: float,
                            snr_min_db: float = -5.0,    # 缩放下限（dB）
                            snr_max_db: float = 35.0,    # 缩放上限（dB）
                            snr_shift_db: Optional[torch.Tensor] = None,  # (B,) 可选域随机化（dB）
                            clamp: bool = False,         # 是否把特征截断到[-1,1]
                            eps: float = 1e-30) -> torch.Tensor:
    """
    功率→SNR特征（按导频子载波）：
      - 直接使用对应信道功率 |h|^2 与噪声功率 sigma2 求 SNR；
      - 在对数域取 dB，再线性映射到 [-1, 1]。
    返回：feat ∈ R^{B×P}，每列对应一个导频子载波的归一化 SNR 特征。
    """
    B, N = H_t.shape
    P = pilot_idx.numel()

    # 选出导频位置的"真实"信道
    h_true = H_t[:, pilot_idx]                              # (B, P)
    pow_lin = (h_true.abs() ** 2).clamp_min(eps)            # |h|^2

    # SNR (linear) 与 dB
    sigma2_lin = max(float(sigma2), eps)
    snr_lin = pow_lin / sigma2_lin
    snr_db = 10.0 * torch.log10(snr_lin.clamp_min(eps))     # (B, P)

    # 可选：dB域平移（域随机化/数据增强）
    if snr_shift_db is not None:
        snr_db = snr_db + snr_shift_db.view(B, 1)

    # 线性映射到 [-1, 1]；默认不截断，超出区间按比例外推
    denom = max(snr_max_db - snr_min_db, 1e-6)
    feat = 2.0 * (snr_db - snr_min_db) / denom - 1.0        # (B, P)

    if clamp:
        feat = feat.clamp_(-1.0, 1.0)

    # 返回实数特征 (B, P)
    return feat


def compute_prior_features(H_t: torch.Tensor,
                           sigma2_lin: float,
                           d_pi: int,
                           snr_min_db: float,
                           snr_max_db: float,
                           noise_std_db: float,
                           eps: float = 1e-30) -> torch.Tensor:
    """Compute prior vector from average subcarrier power."""
    if d_pi <= 0:
        return H_t.new_zeros(H_t.shape[0], 0)
    B = H_t.shape[0]
    avg_pow = H_t.abs().pow(2).mean(dim=1).clamp_min(eps)
    snr_lin = avg_pow * (P_DAT / max(sigma2_lin, eps))
    snr_db = 10.0 * torch.log10(snr_lin)
    if noise_std_db > 0.0:
        snr_db = snr_db + torch.randn_like(snr_db) * noise_std_db

    denom = max(snr_max_db - snr_min_db, 1e-6)
    prior_scalar = 2.0 * (snr_db - snr_min_db) / denom - 1.0
    prior_vec = prior_scalar.unsqueeze(1).expand(B, d_pi)
    return prior_vec


def reward_throughput(H_t: torch.Tensor,
                      action_M: torch.Tensor,
                      data_idx: torch.Tensor,
                      sigma2_db: float,
                      snr_shift_db: Optional[torch.Tensor] = None,
                      use_smooth: bool = True,
                      tau_db: float = 2.0,
                      include_pilots_in_rate: bool = True,
                      pilot_idx: Optional[torch.Tensor] = None,
                      pilot_power_db: Optional[float] = None) -> torch.Tensor:
    """
    计算速率（bit/s）。
    - include_pilots_in_rate=False（默认，保持原逻辑）：只在数据子载波上计算SE，并按 N 做归一化（导频相当于0速率）。
    - include_pilots_in_rate=True：在所有子载波（含导频）上计算SE；可用 pilot_power_db 指定导频功率（dB）。
    """
    B, Ntot = H_t.shape

    # 每批次的门限与比特/符号
    thresholds_db = torch.tensor([MCS_THRESH_DB[int(m)] for m in action_M.tolist()],
                                 device=H_t.device)
    bits_per_sym = torch.log2(action_M.float())  # (B,)

    # 数据子载波功率（dB）
    p_dat_db = 10.0 * math.log10(max(P_DAT, 1e-12))

    if include_pilots_in_rate:
        # --- 全频（含导频）计算 ---
        # |h|^2 → dB
        pow_db_all = power_to_db(H_t.abs() ** 2)               # (B, N)

        # 构造每个子载波的“发射功率 dB”矩阵，默认等于 P_DAT
        p_db_all = torch.full((B, Ntot), p_dat_db, device=H_t.device)

        # 若提供 pilot_power_db，则为导频位置覆写功率（例如 10*log10(P_PIL)）
        if (pilot_idx is not None) and (pilot_power_db is not None):
            p_db_all[:, pilot_idx] = pilot_power_db

        # SNR(dB)
        snr_db = pow_db_all + p_db_all - sigma2_db
        if snr_shift_db is not None:
            snr_db = snr_db + snr_shift_db.view(B, 1)

        # 门控（平滑/硬判）
        if use_smooth:
            gate = torch.sigmoid((snr_db - thresholds_db.view(B, 1)) / tau_db)
        else:
            gate = (snr_db >= thresholds_db.view(B, 1)).float()

        # SE 与速率：此时是对 N 个子载波取平均
        se = R_C * bits_per_sym.view(B, 1) * gate              # (B, N)
        se_mean = se.mean(dim=1)                               # 已含全频平均
        R = W * se_mean                                        # (B,)
        return R
    else:
        # --- 原有：仅数据子载波 ---
        Hd = H_t[:, data_idx]                                  # (B, Nd)
        pow_db = power_to_db(Hd.abs() ** 2)
        snr_db = pow_db + p_dat_db - sigma2_db                 # (B, Nd)
        if snr_shift_db is not None:
            snr_db = snr_db + snr_shift_db.view(B, 1)

        if use_smooth:
            gate = torch.sigmoid((snr_db - thresholds_db.view(B, 1)) / tau_db)
        else:
            gate = (snr_db >= thresholds_db.view(B, 1)).float()

        se = R_C * bits_per_sym.view(B, 1) * gate              # (B, Nd)
        # 仍按全 N 归一化（导频=0 的口径）
        se_mean = se.mean(dim=1) * (Hd.shape[1] / Ntot)
        R = W * se_mean
        return R


# -----------------------------
# External dataset helpers
# -----------------------------


def _load_mat_variable(mat_path: str, var_name: str):
    """Load a MATLAB variable supporting both v7 and v7.3 files."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    data = None
    if sio is not None:
        try:
            mat_dict = sio.loadmat(mat_path)
            if var_name in mat_dict:
                data = mat_dict[var_name]
        except Exception:
            data = None

    if data is None:
        import h5py
        with h5py.File(mat_path, "r") as f:
            if var_name not in f:
                raise KeyError(f'Variable "{var_name}" not found in {mat_path}')
            data = np.array(f[var_name]).transpose()
    return np.array(data)


def load_mat_channel_dataset(mat_path: str,
                             var_name: str = "H_cellfree_time",
                             val_ratio: float = 0.2,
                             device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load external cell-free channel tensor saved via save(output_path, 'H_cellfree_time', '-v7.3');
    reshape to (B, L, N) with B = nSim*num_BS*num_K*num_TxAnt, L = NumSnapshots, N = num_subcarriers.
    Returns train and validation tensors on the requested device.
    """
    H = _load_mat_variable(mat_path, var_name)
    if H.ndim != 6:
        raise ValueError(f'Expected 6-D tensor for "{var_name}", got shape {H.shape}')

    nSim, num_snapshots, num_bs, num_k, num_tx_ant, num_subcarriers = H.shape
    if nSim <= 0 or num_snapshots <= 0 or num_subcarriers <= 0:
        raise ValueError(f"Invalid dimensions in loaded tensor: {H.shape}")

    if H.dtype.fields is not None and set(H.dtype.fields.keys()) == {"real", "imag"}:
        H = H["real"] + 1j * H["imag"]       # 合成复数
    H = H.astype(np.complex64, copy=False)
    H = H.reshape(nSim, num_snapshots, num_bs * num_k * num_tx_ant, num_subcarriers)
    H = np.transpose(H, (0, 2, 1, 3))  # -> (nSim, B_per_sim, L, N)
    H = H.reshape(nSim * num_bs * num_k * num_tx_ant, num_snapshots, num_subcarriers)

    total_B = H.shape[0]
    if total_B == 0:
        raise ValueError("Loaded dataset is empty after reshaping.")

    perm = np.random.permutation(total_B)
    val_ratio = float(min(max(val_ratio, 0.0), 1.0))

    if total_B == 1:
        train_idx = perm
        val_idx = perm
    else:
        val_count = int(round(total_B * val_ratio))
        val_count = min(max(val_count, 1), total_B - 1)
        val_idx = perm[:val_count]
        train_idx = perm[val_count:]

    H_torch = torch.from_numpy(H)
    train_tensor = H_torch.index_select(0, torch.as_tensor(train_idx, dtype=torch.long))
    val_tensor = H_torch.index_select(0, torch.as_tensor(val_idx, dtype=torch.long))
    return train_tensor.to(device), val_tensor.to(device)


# -----------------------------
# Models
# -----------------------------
class UENet(nn.Module):
    def __init__(self, P: int, d_hidden: int, d_out: int):
        super().__init__()
        in_dim = P
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
            nn.ReLU(),
        )
    def forward(self, pilot_feats: torch.Tensor) -> torch.Tensor:
        return self.net(pilot_feats)

class FeatureBlock(nn.Module):
    """Small helper MLP used to keep BS input dimension consistent across modes."""
    def __init__(self, in_dim: int, d_hidden: int, d_out: int):
        super().__init__()
        if in_dim <= 0:
            raise ValueError("FeatureBlock requires in_dim > 0")
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BSNetPolicy(nn.Module):
    def __init__(self,
                 d_ue: int,
                 d_pi: int,
                 d_hidden: int,
                 num_actions: int,
                 prior_mode: str = "both"):
        super().__init__()
        self.prior_mode = prior_mode
        self.d_fused = d_ue + max(d_pi, 0)
        self._route_behavior = "learned"

        self.pilot_block = FeatureBlock(d_ue, d_hidden, self.d_fused)
        self.prior_block = FeatureBlock(d_pi, d_hidden, self.d_fused) if d_pi > 0 else None
        self.both_block = FeatureBlock(d_ue + d_pi, d_hidden, self.d_fused)

        if prior_mode == "route":
            if d_pi <= 0:
                raise ValueError("route mode requires D_PI > 0 for prior features.")
            gate_in_dim = d_ue + d_pi
            self.route_gate = nn.Sequential(
                nn.Linear(gate_in_dim, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, 2),
            )
        else:
            self.route_gate = None

        print(self.d_fused)

        self.head = nn.Sequential(
            nn.Linear(self.d_fused, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, num_actions),
        )
        self.register_buffer(
            "route_costs",
            torch.tensor([ROUTE_MODE_A_COST, ROUTE_MODE_B_COST], dtype=torch.float32),
        )

    def forward(self, ue_embed: torch.Tensor, prior_vec: torch.Tensor):
        fused, aux = self._mix_features(ue_embed, prior_vec)
        logits = self.head(fused)
        return logits, aux

    def set_route_behavior(self, behavior: str):
        """Set gating behavior for route mode: 'learned' or 'uniform'."""
        if behavior not in {"learned", "uniform"}:
            raise ValueError("behavior must be 'learned' or 'uniform'")
        self._route_behavior = behavior

    def gate_parameters(self):
        if self.route_gate is None:
            return []
        return list(self.route_gate.parameters())

    def non_gate_parameters(self):
        params = []
        for module in (self.pilot_block, self.prior_block, self.both_block, self.head):
            if module is None:
                continue
            params.extend(list(module.parameters()))
        return params

    def _mix_features(self, ue_embed: torch.Tensor, prior_vec: torch.Tensor):
        aux = None
        if self.prior_mode == "pilot_only":
            fused = self.pilot_block(ue_embed)
        elif self.prior_mode == "prior_only":
            if self.prior_block is None:
                raise ValueError("prior_only mode requires D_PI > 0.")
            fused = self.prior_block(prior_vec)
        elif self.prior_mode == "both":
            fused_input = torch.cat([ue_embed, prior_vec], dim=-1)
            fused = self.both_block(fused_input)
        elif self.prior_mode == "route":
            if self.prior_block is None:
                raise ValueError("route mode requires prior features.")
            combined = torch.cat([ue_embed, prior_vec], dim=-1)
            behavior = getattr(self, "_route_behavior", "learned")
            if behavior == "uniform":
                B = combined.shape[0]
                gate_probs = torch.full((B, 2), 0.5, device=combined.device, dtype=combined.dtype)
                rand_idx = torch.randint(0, 2, (B,), device=combined.device)
                gate_hard = F.one_hot(rand_idx, num_classes=2).float()
            else:
                gate_logits = self.route_gate(combined)
                gate_probs = torch.softmax(gate_logits, dim=-1)
                if self.training:
                    gate_hard = F.gumbel_softmax(gate_logits, tau=ROUTE_GUMBEL_TAU, hard=True)
                else:
                    gate_idx = torch.argmax(gate_probs, dim=-1)
                    gate_hard = F.one_hot(gate_idx, num_classes=2).float()
            feat_a = self.prior_block(prior_vec)
            feat_b = self.both_block(combined)
            fused = gate_hard[:, 0:1] * feat_a + gate_hard[:, 1:2] * feat_b
            aux = {
                "gate_probs": gate_probs,
                "gate_hard": gate_hard,
            }
        else:
            raise ValueError(f"Unsupported PRIOR_MODE '{self.prior_mode}'")
        return fused, aux


def log_model_parameters(label: str, model: nn.Module) -> int:
    """Print parameter details and return total count."""
    print(f"\n=== {label} Parameters ===")
    total = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        shape = "x".join(str(dim) for dim in param.shape)
        print(f"{label}.{name:<35} shape=({shape}) count={count:,}")
    print(f"Total {label} params: {total / 1e6:.4f} M ({total:,})\n")
    return total


# -----------------------------
# Training helpers
# -----------------------------
@dataclass
class TrainState:
    ue: UENet
    bs: BSNetPolicy
    opt: optim.Optimizer

@dataclass
class EpochMetrics:
    epoch: int
    train_rate: float
    val_rate: float
    action_dist_train: list
    action_dist_val: list

class MetricsLogger:
    def __init__(self):
        self._history = []

    def log_epoch(self, epoch, train_rate, val_rate, action_dist_train, action_dist_val):
        self._history.append(
            EpochMetrics(
                epoch=epoch,
                train_rate=train_rate,
                val_rate=val_rate,
                action_dist_train=list(action_dist_train),
                action_dist_val=list(action_dist_val),
            )
        )

    @property
    def epochs(self):
        return [m.epoch for m in self._history]

    @property
    def train_rates(self):
        return [m.train_rate for m in self._history]

    @property
    def val_rates(self):
        return [m.val_rate for m in self._history]

    def action_matrix(self, which="train"):
        if which == "train":
            data = [m.action_dist_train for m in self._history]
        elif which == "val":
            data = [m.action_dist_val for m in self._history]
        else:
            raise ValueError("which must be 'train' or 'val'")
        return np.array(data)


def set_requires_grad(params, requires_grad: bool):
    for p in params:
        p.requires_grad = requires_grad


def configure_optimizer_for_stage(state: TrainState, stage: str) -> optim.Optimizer:
    ue, bs = state.ue, state.bs
    ue_params = list(ue.parameters())
    bs_gate_params = bs.gate_parameters()
    bs_non_gate_params = bs.non_gate_parameters()

    if stage == "pre_expert":
        bs.set_route_behavior("uniform")
        train_params = ue_params + bs_non_gate_params
        set_requires_grad(train_params, True)
        set_requires_grad(bs_gate_params, False)
    elif stage == "gate_only":
        bs.set_route_behavior("learned")
        train_params = bs_gate_params
        set_requires_grad(ue_params + bs_non_gate_params, False)
        set_requires_grad(bs_gate_params, True)
    else:
        bs.set_route_behavior("learned")
        train_params = ue_params + bs_non_gate_params + bs_gate_params
        set_requires_grad(train_params, True)

    if not train_params:
        raise ValueError(f"No parameters available for stage '{stage}'.")

    state.opt = optim.Adam(train_params, lr=LR)
    return state.opt

def build_models(P: int, d_u: int, d_pi: int, hidden_ue: int, hidden_bs: int, num_actions: int) -> TrainState:
    ue = UENet(P, hidden_ue, d_u).to(DEVICE)
    bs = BSNetPolicy(d_u, d_pi, hidden_bs, num_actions, prior_mode=PRIOR_MODE).to(DEVICE)
    params = list(ue.parameters()) + list(bs.parameters())
    opt = optim.Adam(params, lr=LR)
    return TrainState(ue=ue, bs=bs, opt=opt)

def batchify(H: torch.Tensor, batch_size: int):
    B_total = H.shape[0]
    idx = torch.randperm(B_total, device=H.device)
    for i in range(0, B_total, batch_size):
        sel = idx[i:i+batch_size]
        yield H[sel]

def evaluate(state: TrainState,
             H_val: torch.Tensor,
             pilot_idx: torch.Tensor,
             data_idx: torch.Tensor) -> Dict[str, float]:
    ue, bs = state.ue, state.bs
    ue.eval(); bs.eval()
    mcs_tensor = torch.tensor(MCS_LIST, device=H_val.device)

    total_reward_sum = 0.0
    action_hist = torch.zeros(len(MCS_LIST), device=H_val.device)

    with torch.no_grad():
        for Hb in batchify(H_val, BATCH_SIZE):
            B, Lb, _ = Hb.shape
            for t in range(Lb):
                H_t = Hb[:, t, :]
                pilot_feats = ls_estimate_from_pilots(H_t, pilot_idx, P_PIL, SIGMA2_LIN, snr_shift_db=None)
                z_ue = ue(pilot_feats)
                prior_vec = torch.zeros((B, D_PI), device=H_t.device)
                if D_PI > 0 and PRIOR_MODE != "pilot_only":
                    prior_vec = compute_prior_features(
                        H_t, SIGMA2_LIN, D_PI,
                        PRIOR_SNR_MIN_DB, PRIOR_SNR_MAX_DB,
                        noise_std_db=PRIOR_NOISE_STD_DB)
                if PRIOR_MODE == "prior_only":
                    z_ue = torch.zeros_like(z_ue)
                elif PRIOR_MODE == "pilot_only":
                    prior_vec = torch.zeros_like(prior_vec)

                logits, _ = bs(z_ue, prior_vec)
                a_idx = torch.argmax(logits, dim=-1)
                M = mcs_tensor[a_idx]
                r = reward_throughput(H_t, M, data_idx, SIGMA2_DB, snr_shift_db=None,
                                      use_smooth=USE_SMOOTH_REWARD, tau_db=TAU_DB)

                total_reward_sum += r.sum().item()

                counts = torch.bincount(a_idx, minlength=len(MCS_LIST)).float()
                action_hist += counts

    num_sequences = max(1, H_val.shape[0])
    num_time_steps = max(1, H_val.shape[1])
    avg_reward_per_sequence = total_reward_sum / float(num_sequences)
    rate = avg_reward_per_sequence / float(num_time_steps)
    action_hist = (action_hist / action_hist.sum().clamp_min(1.0)).tolist()
    return {
        "rate": rate,
        "return": rate,
        "action_dist_greedy": action_hist,
    }

# -----------------------------
# Training loop (REINFORCE)
# -----------------------------
def train_loop():
    set_seed(SEED)

    if DATA_SOURCE.lower() == "mat":
        H_train, H_val = load_mat_channel_dataset(MAT_DATA_PATH,
                                                  var_name=MAT_VAR_NAME,
                                                  val_ratio=MAT_VAL_RATIO,
                                                  device=DEVICE)
    else:
        pass

    num_subcarriers = H_train.shape[-1]
    pilot_idx = get_pilot_indices(num_subcarriers, PILOT_STRIDE).to(DEVICE)
    all_idx   = torch.arange(num_subcarriers, device=DEVICE)
    data_mask = torch.ones(num_subcarriers, dtype=torch.bool, device=DEVICE); data_mask[pilot_idx] = False
    data_idx  = all_idx[data_mask]

    state = build_models(P=pilot_idx.numel(), d_u=D_U, d_pi=D_PI,
                         hidden_ue=HIDDEN_UE, hidden_bs=HIDDEN_BS,
                         num_actions=len(MCS_LIST))
    ue, bs = state.ue, state.bs
    opt = state.opt
    mcs_tensor  = torch.tensor(MCS_LIST, device=DEVICE)
    metrics_logger = MetricsLogger()

    total_ue = log_model_parameters("UE", ue)
    total_bs = log_model_parameters("BS", bs)
    print(f"UE params: {total_ue / 1e6:.4f} M | BS params: {total_bs / 1e6:.4f} M | "
          f"Combined: {(total_ue + total_bs) / 1e6:.4f} M\n")

    print(f"Data source: {DATA_SOURCE} | Train set: {H_train.shape}, Val set: {H_val.shape}, "
          f"pilots: {pilot_idx.numel()}, data tones: {data_idx.numel()}")
    print(f"Actions (MCS): {MCS_LIST} with thresholds (dB): {MCS_THRESH_DB}")
    print(f"Reward smoothing: {USE_SMOOTH_REWARD} (tau_db={TAU_DB}), epsilon={EPSILON}, entropy_coef={ENTROPY_COEF}, SNR jitter +/-{SNR_JITTER_DB} dB\n")

    total_epochs = EPOCHS if PRIOR_MODE != "route" else (EPOCH_PRE + EPOCH_PRE_GATE)
    if total_epochs <= 0:
        raise ValueError("Total number of training epochs must be positive.")

    def epoch_stage(epoch_idx: int) -> str:
        if PRIOR_MODE != "route":
            return "standard"
        return "pre_expert" if epoch_idx <= EPOCH_PRE else "gate_only"

    current_stage = None

    for epoch in range(1, total_epochs + 1):
        stage = epoch_stage(epoch)
        if stage != current_stage:
            opt = configure_optimizer_for_stage(state, stage)
            current_stage = stage
            if PRIOR_MODE == "route":
                print(f"\n[Stage Switch] Entering {stage} phase at epoch {epoch}")

        ue.train(); bs.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_infer_loss = 0.0
        total_steps = 0
        action_hist = torch.zeros(len(MCS_LIST), device=DEVICE)
        track_route = (PRIOR_MODE == "route")
        route_prob_sums = torch.zeros(2, device=DEVICE) if track_route else None
        route_prob_count = 0
        apply_infer_penalty = (PRIOR_MODE == "route" and stage == "gate_only")

        # Accumulate total reward over the training set
        epoch_reward_sum = 0.0

        for Hb in batchify(H_train, BATCH_SIZE):
            B, Lb, _ = Hb.shape

            log_probs = []
            entropies = []
            rewards = []
            gate_probs_buffer = [] if PRIOR_MODE == "route" else None

            for t in range(Lb):
                H_t = Hb[:, t, :]

                # Apply per-TTI SNR jitter during training
                if SNR_JITTER_DB > 0.0:
                    snr_shift_db = torch.empty(B, device=DEVICE).uniform_(-SNR_JITTER_DB, SNR_JITTER_DB)
                else:
                    snr_shift_db = None

                pilot_feats = ls_estimate_from_pilots(H_t, pilot_idx, P_PIL, SIGMA2_LIN, snr_shift_db=snr_shift_db)
                z_ue = ue(pilot_feats)

                prior_vec = torch.zeros((B, D_PI), device=H_t.device)
                if D_PI > 0 and PRIOR_MODE != "pilot_only":
                    prior_vec = compute_prior_features(
                        H_t, SIGMA2_LIN, D_PI,
                        PRIOR_SNR_MIN_DB, PRIOR_SNR_MAX_DB,
                        noise_std_db=PRIOR_NOISE_STD_DB)
                if PRIOR_MODE == "prior_only":
                    z_ue = torch.zeros_like(z_ue)
                elif PRIOR_MODE == "pilot_only":
                    prior_vec = torch.zeros_like(prior_vec)

                # logits = bs(z_ue, prior) / max(1e-6, TEMPERATURE)
                # dist = Categorical(logits=logits)

                # # Epsilon-greedy applied on top of softmax
                # a_idx = dist.sample()
                # if EPSILON > 0.0:
                #     rand_mask = (torch.rand(B, device=DEVICE) < EPSILON)
                #     rand_actions = torch.randint(low=0, high=len(MCS_LIST), size=(B,), device=DEVICE)
                #     a_idx = torch.where(rand_mask, rand_actions, a_idx)

                # logp = dist.log_prob(a_idx)
                # ent  = dist.entropy()
                logits, aux = bs(z_ue, prior_vec)
                logits = logits / max(1e-6, TEMPERATURE)
                if PRIOR_MODE == "route" and aux is not None and gate_probs_buffer is not None:
                    gate_probs = aux["gate_probs"]
                    gate_probs_buffer.append(gate_probs)
                    if route_prob_sums is not None:
                        route_prob_sums += gate_probs.detach().sum(dim=0)
                        route_prob_count += gate_probs.shape[0]
                probs  = torch.softmax(logits, dim=-1)
                num_a  = probs.size(-1)
                probs_mix = (1.0 - EPSILON) * probs + EPSILON / num_a
                dist_mix  = Categorical(probs=probs_mix)

                a_idx = dist_mix.sample()
                logp  = dist_mix.log_prob(a_idx)
                ent   = -(probs_mix * (probs_mix.clamp_min(1e-12)).log()).sum(dim=-1)  # Entropy consistent with the mixed policy

                M = mcs_tensor[a_idx]
                r = reward_throughput(H_t, M, data_idx, SIGMA2_DB, snr_shift_db=snr_shift_db,
                                      use_smooth=USE_SMOOTH_REWARD, tau_db=TAU_DB)

                log_probs.append(logp)
                entropies.append(ent)
                rewards.append(r)

                with torch.no_grad():
                    counts = torch.bincount(a_idx, minlength=len(MCS_LIST)).float()
                    action_hist += counts
                    epoch_reward_sum += r.sum().item()

            # Stack across the time dimension
            log_probs_t = torch.stack(log_probs, dim=0)   # (L, B)
            entropies_t = torch.stack(entropies, dim=0)   # (L, B)
            rewards_t   = torch.stack(rewards, dim=0)     # (L, B)

            # Discounted returns
            returns = torch.zeros_like(rewards_t)
            G = torch.zeros((B,), device=DEVICE)
            for t in reversed(range(Lb)):
                G = rewards_t[t] + DISCOUNT * G
                returns[t] = G

            # Baseline plus advantage normalization
            baseline    = returns.mean()
            advantages  = returns - baseline
            adv_mean    = advantages.mean()
            adv_std     = advantages.std().clamp_min(1e-6)
            advantages  = (advantages - adv_mean) / adv_std

            # Loss
            policy_loss  = -(log_probs_t * advantages.detach()).mean()
            entropy_loss = -ENTROPY_COEF * entropies_t.mean()
            infer_loss = torch.zeros((), device=DEVICE)
            if apply_infer_penalty and gate_probs_buffer:
                gate_probs_t = torch.stack(gate_probs_buffer, dim=0)  # (L, B, 2)
                cost_tensor = bs.route_costs.view(1, 1, -1)
                infer_cost = torch.sum(gate_probs_t * cost_tensor, dim=-1)
                infer_loss = INFER_LAMBDA * infer_cost.mean()
            loss = policy_loss + entropy_loss + infer_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(list(ue.parameters()) + list(bs.parameters()), GRAD_CLIP)
            opt.step()

            total_loss         += loss.item()
            total_policy_loss  += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_infer_loss   += infer_loss.item()
            total_steps        += 1

        # Training statistics
        num_train_sequences = max(1, H_train.shape[0])
        num_time_steps = max(1, H_train.shape[1])
        avg_reward_per_sequence = epoch_reward_sum / float(num_train_sequences)
        train_return = avg_reward_per_sequence / float(num_time_steps)
        train_rate = train_return
        action_hist = action_hist / action_hist.sum().clamp_min(1.0)

        # Validation
        eval_stats = evaluate(state, H_val, pilot_idx, data_idx)

        ah_str   = ", ".join([f"M{MCS_LIST[i]}:{action_hist[i].item():.2f}" for i in range(len(MCS_LIST))])
        ah_eval  = ", ".join([f"M{MCS_LIST[i]}:{eval_stats['action_dist_greedy'][i]:.2f}" for i in range(len(MCS_LIST))])
        avg_policy_loss = total_policy_loss / max(1, total_steps)
        avg_entropy_loss = total_entropy_loss / max(1, total_steps)
        avg_infer_loss = total_infer_loss / max(1, total_steps)
        avg_total_loss = total_loss / max(1, total_steps)
        route_msg = ""
        if track_route and route_prob_sums is not None and route_prob_count > 0:
            route_probs = (route_prob_sums / route_prob_count).tolist()
            route_msg = f" | route_p=(A:{route_probs[0]:.2f}, B:{route_probs[1]:.2f})"
        print(f"[Epoch {epoch:02d}] stage={stage} "
              f"avd={torch.abs(advantages).mean().item():.4f} | log_probs_t={log_probs_t.mean().item():.4f} | "
              f"train_rate={train_rate/1e6:.3f} Mbps | val_rate={eval_stats['rate']/1e6:.3f} Mbps | "
              f"act_train=({ah_str}) | act_val(greedy)=({ah_eval}) | "
              f"policy loss={avg_policy_loss:.4f} | entropy loss={avg_entropy_loss:.4f} | "
              f"infer loss={avg_infer_loss:.4f} | total loss={avg_total_loss:.4f}{route_msg}")
        metrics_logger.log_epoch(
            epoch=epoch,
            train_rate=train_rate,
            val_rate=eval_stats["rate"],
            action_dist_train=action_hist.tolist(),
            action_dist_val=eval_stats["action_dist_greedy"],
        )

    plot_training_curves(metrics_logger, action_labels=MCS_LIST, show=True)

def plot_training_curves(logger: MetricsLogger,
                         action_labels,
                         show: bool = True,
                         save_prefix: Optional[str] = None,
                         include_validation: bool = True,
                         subplot_height_ratio: Tuple[float, float] = (0.5, 1.0),
                         figure_size: Tuple[float, float] = (8.0, 5)):
    if not logger.epochs:
        return

    epochs = np.array(logger.epochs)
    train_rates = np.array(logger.train_rates) / 1e6  # Convert to Mbps
    val_rates = np.array(logger.val_rates) / 1e6

    with suppress(OSError, ValueError):
        plt.style.use("seaborn-v0_8")

    ratio_top, ratio_bottom = subplot_height_ratio
    fig_train, (ax_train_top, ax_train_bottom) = plt.subplots(
        2, 1, figsize=figure_size, sharex=True,
        gridspec_kw={"height_ratios": [ratio_top, ratio_bottom]})
    ax_train_top.plot(epochs, train_rates, linewidth=1.5, markerfacecolor='white', markeredgecolor='blue')
    ax_train_top.set_ylabel("Throughput (Mbps)")
    ax_train_top.set_title("Train Throughput vs. Epoch")
    ax_train_top.grid(True, alpha=0.3)

    def _plot_action_share(axis, which: str, title: str):
        action_share = logger.action_matrix(which)
        if action_share.ndim != 2 or action_share.size == 0:
            return False
        if action_share.shape[0] == 1:
            epsilon = 1e-6
            epochs_stack = np.concatenate([epochs, epochs + epsilon])
            action_share_stack = np.vstack([action_share[0], action_share[0]])
        else:
            epochs_stack = epochs
            action_share_stack = action_share

        cmap = plt.cm.get_cmap("Blues")
        levels = [1.0, 0.9, 0.75, 0.5, 0.35]
        colors = cmap(levels)

        axis.stackplot(epochs_stack, action_share_stack.T, colors=colors, labels=[f"M{m}" for m in action_labels])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Action Probability")
        axis.set_title(title)
        if len(epochs) <= 10:
            axis.set_xticks(epochs)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper center", ncol=min(len(action_labels), 5), bbox_to_anchor=(0.5, -0.2))
        return True

    train_share_plotted = _plot_action_share(ax_train_bottom, "train", "Train Action Distribution")
    if not train_share_plotted:
        fig_train.delaxes(ax_train_bottom)

    fig_val = None
    if include_validation and len(val_rates) > 0:
        fig_val, (ax_val_top, ax_val_bottom) = plt.subplots(
            2, 1, figsize=figure_size, sharex=True,
            gridspec_kw={"height_ratios": [ratio_top, ratio_bottom]})
        ax_val_top.plot(epochs, val_rates, linewidth=1.5)
        ax_val_top.set_ylabel("Throughput (Mbps)")
        ax_val_top.set_title("Validation Throughput vs. Epoch")
        ax_val_top.grid(True, alpha=0.3)
        share_ok = _plot_action_share(ax_val_bottom, "val", "Validation Action Distribution")
        if not share_ok:
            fig_val.delaxes(ax_val_bottom)

    fig_train.tight_layout()
    if fig_val is not None:
        fig_val.tight_layout()

    if save_prefix:
        fig_train.savefig(f"{save_prefix}_train_summary.png", dpi=200)
        if fig_val is not None:
            fig_val.savefig(f"{save_prefix}_val_summary.png", dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig_train)
        if fig_val is not None:
            plt.close(fig_val)

if __name__ == "__main__":
    train_loop()
