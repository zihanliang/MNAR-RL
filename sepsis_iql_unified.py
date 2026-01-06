from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Part 0: Logging & Constants
# =============================================================================

LOGGER = logging.getLogger("sepsis_iql")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(asctime)s | %(message)s", datefmt="%H:%M:%S")
    _h.setFormatter(_fmt)
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

# Expected dimensions from processed_data_v2 (metadata.json)
EXPECTED_T = 73          # 1h observation grid (0-72h)
EXPECTED_K = 18          # 4h decision steps
EXPECTED_D = 16          # observation features (8 vitals + 8 labs)
EXPECTED_S = 3           # static features (age, gender_male, charlson_index)

# IMPORTANT:
# text_dim is dataset-dependent (e.g., 256 for hashing, 768 for BERT).
# We treat this as a DEFAULT fallback only; actual value comes from metadata.json.
DEFAULT_TEXT_DIM = 256

EXPECTED_N_ACTIONS = 9   # 3 fluids × 3 vaso

# Required arrays - a_1h is NOT required since we only use 4h decisions
REQUIRED_ARRAY_KEYS = (
    "stay_id", "hadm_id", "subject_id", "X_static",
    "t_1h", "Y", "mask", "delta", "time_mask",
    "t_4h", "a_4h", "r_4h", "done_4h", "step_mask_4h",
    "e_rad", "e_micro", "m_text",
)

# Optional arrays (not required for training)
OPTIONAL_ARRAY_KEYS = (
    "a_1h",  # moved here - not used in training
    "fluids_ml_4h",
    "vaso_ne_4h",
)


# =============================================================================
# Part 1: Configuration Dataclasses
# =============================================================================

@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: str = "./processed_data_v2"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    mmap: bool = True
    normalize_obs: bool = True
    normalize_static: bool = True
    include_exposures: bool = False
    compute_mnar_features: bool = True
    mnar_window_hours: int = 6


@dataclass
class EncoderConfig:
    """Encoder architecture configuration."""
    # GRU-D
    d_hidden_1h: int = 128
    d_delta_embed: int = 16
    d_time_embed: int = 16
    gru_dropout: float = 0.1

    # === MNAR feature fusion ===
    # mnar_feat at 1h grid has dim = D*4 (delta, cumcount, missrate, windowfreq)
    enable_mnar_fusion: bool = True
    d_mnar_embed: int = 32

    # Decision state
    d_state: int = 128
    state_mode: str = "at_decision"  # "at_decision" or "block_pool"

    # Text fusion
    enable_text_fusion: bool = True
    text_n_heads: int = 4
    text_dropout: float = 0.1

    # Latent dynamics (optional)
    enable_latent_dynamics: bool = False
    d_latent: int = 32
    dynamics_hidden: int = 64


@dataclass
class IQLConfig:
    """
    Pure IQL configuration.

    Core IQL:
    1) Q-TD: MSE(Q(s,a), r + γ(1-d)V_target(s'))
    2) V-expectile: E[|τ - 1(δ<0)| · δ²] where δ = Q_min(s,a) - V(s)
    3) π-AWBC: -E[clip(exp(A/β), w_max) · log π(a|s)]

    Optional stabilization:
    - use_target_q: maintain Polyak target Q networks to compute Q_min for V/Adv (default False).
      This does NOT change the IQL objective form, only reduces variance / instability.
    """
    # Core IQL
    expectile: float = 0.7
    awbc_beta: float = 3.0
    awbc_clip: float = 20.0
    gamma: float = 0.99
    tau: float = 0.005  # Polyak for targets (V and optionally Q)

    # Optional target-Q stabilizer
    use_target_q: bool = False

    # Advantage processing
    adv_normalize: bool = True

    # Q-value handling (for terminal-only reward)
    use_huber_loss: bool = True
    huber_delta: float = 1.0
    td_target_clip: float = 2.0

    # === STABILIZERS (optional, for ablation) ===
    use_entropy_stabilizer: bool = False
    entropy_stabilizer_coef: float = 0.01
    entropy_stabilizer_min: float = 0.3  # nats

    # Numerical stability only (NOT policy smoothing!)
    log_prob_floor: float = 1e-8
    logit_clip: float = 10.0


@dataclass
class TrainConfig:
    """Training configuration with anti-collapse mechanisms."""
    # Basic training
    total_steps: int = 200000
    lr_critic: float = 3e-4
    lr_value: float = 3e-4
    lr_actor: float = 1e-4           # smaller for actor
    lr_encoder: float = 3e-4
    lr_aux: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    # === Anti-collapse: Training schedule ===
    critic_warmup_steps: int = 10000   # only train Q/V, no actor
    actor_update_every: int = 2        # TD3-style delayed actor

    # === Anti-collapse: Encoder freezing strategy ===
    encoder_pretrain_steps: int = 5000
    encoder_freeze_steps: int = 50000
    encoder_finetune_lr_scale: float = 0.1

    # === Anti-collapse: Auto-detection & rollback ===
    collapse_entropy_threshold: float = 0.3   # nats
    collapse_top1_threshold: float = 0.95
    collapse_patience: int = 3
    rollback_lr_scale: float = 0.5
    rollback_beta_scale: float = 1.5
    rollback_clip_scale: float = 0.7

    # Evaluation & checkpointing
    eval_every: int = 2000
    save_every: int = 10000
    log_every: int = 100

    # === Cost control: FQE during training ===
    fqe_eval_every: int = 10000
    fqe_steps_during_training: int = 5000

    # === Early stopping (based on FQE-fast LCB) ===
    enable_early_stop: bool = True
    early_stop_min_steps: int = 60000 
    early_stop_patience: int = 5
    early_stop_delta: float = 1e-4

    # Seed
    seed: int = 42


@dataclass
class AuxLossConfig:
    """Auxiliary loss configuration (representation learning)."""
    enable_reconstruction: bool = False
    w_recon_obs: float = 0.1
    w_recon_mask: float = 0.05
    w_recon_text: float = 0.05
    
    # Outcome prediction (mortality) - this one works correctly
    enable_outcome: bool = True
    w_outcome: float = 0.1
    
    # Latent KL (if using latent dynamics)
    enable_latent_kl: bool = False   # renamed from enable_kl to avoid confusion with policy KL
    w_latent_kl: float = 0.01
    
    # Dynamics consistency (if using action-conditioned dynamics)
    enable_dynamics: bool = False
    w_dynamics: float = 0.1


@dataclass
class OPEConfig:
    """Off-Policy Evaluation configuration (AI Clinician aligned)."""
    # Behavior policy estimation
    bc_epochs: int = 3
    bc_lr: float = 1e-3
    
    # WIS configuration
    soften_eps: float = 0.01         # 99%/1% policy softening
    ratio_clip: float = 50.0         # truncate importance ratios
    
    # Bootstrap
    bootstrap_n: int = 2000
    bootstrap_alpha: float = 0.05    # 95% CI
    
    # FQE configuration
    fqe_steps: int = 20000
    fqe_lr: float = 3e-4
    fqe_target_tau: float = 0.005
    
    # Reporting
    report_ess: bool = True
    report_dose_gap: bool = True


@dataclass
class ExperimentConfig:
    """Master configuration."""
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    iql: IQLConfig = field(default_factory=IQLConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aux: AuxLossConfig = field(default_factory=AuxLossConfig)
    ope: OPEConfig = field(default_factory=OPEConfig)
    
    # Run management
    run_name: str = ""
    run_dir: str = "./runs"
    device: str = "cuda"
    
    def __post_init__(self):
        if not self.run_name:
            self.run_name = f"iql_exp{self.iql.expectile}_beta{self.iql.awbc_beta}_{int(time.time())}"


# =============================================================================
# Part 2: Data Loading
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int) -> None:
    """
    Ensure each DataLoader worker has a deterministic (but different) seed.

    This makes numpy/random transforms stable when num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_np_load(path: Path, mmap: bool) -> np.ndarray:
    return np.load(str(path), mmap_mode="r" if mmap else None, allow_pickle=False)


@dataclass(frozen=True)
class DataSpec:
    """Dataset specification inferred from metadata."""
    n_stays: int
    T: int
    K: int
    D: int
    S: int
    text_dim: int
    n_actions: int
    obs_feature_names: Tuple[str, ...]
    static_feature_names: Tuple[str, ...]
    shaping_alpha: float


def load_metadata(data_dir: Path) -> Dict[str, Any]:
    """Load and validate metadata.json."""
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    return _read_json(meta_path)


def infer_spec(md: Dict[str, Any]) -> DataSpec:
    """Infer DataSpec from metadata."""
    n_stays = int(md["n_stays"])
    cfg = md.get("config", {})
    
    horizon = int(md.get("horizon_hours", cfg.get("horizon_hours", 72)))
    obs_dt = int(md.get("obs_dt_hours", cfg.get("obs_dt_hours", 1)))
    dec_dt = int(md.get("decision_dt_hours", cfg.get("decision_dt_hours", 4)))
    
    T = horizon // obs_dt + 1
    K = horizon // dec_dt
    
    obs_names = tuple(md["obs_feature_names"])
    static_names = tuple(md["static_feature_names"])
    D = len(obs_names)
    S = len(static_names)
    
    text_dim = int(md.get("text_dim", cfg.get("text_dim", 256)))
    n_actions = int(md["action_space"]["n_actions"])
    shaping = float(md.get("reward", {}).get("shaping_alpha", 0.0))
    
    return DataSpec(
        n_stays=n_stays, T=T, K=K, D=D, S=S,
        text_dim=text_dim, n_actions=n_actions,
        obs_feature_names=obs_names,
        static_feature_names=static_names,
        shaping_alpha=shaping,
    )


class ProcessedDataStore:
    """Lazy memmap loader for processed_data_v2 arrays."""
    
    def __init__(self, data_dir: Union[str, Path], mmap: bool = True):
        self.data_dir = _as_path(data_dir)
        self.mmap = mmap
        self.metadata = load_metadata(self.data_dir)
        self.spec = infer_spec(self.metadata)
        self._arrays: Dict[str, np.ndarray] = {}
        self._validate()
    
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_arrays"] = {}  # don't pickle memmaps
        return d
    
    def _validate(self):
        """Check required arrays exist."""
        for k in REQUIRED_ARRAY_KEYS:
            p = self.data_dir / f"{k}.npy"
            if not p.exists():
                raise FileNotFoundError(f"Missing {k}.npy in {self.data_dir}")
    
    def get(self, key: str) -> np.ndarray:
        """Get array by key (lazy load)."""
        if key in self._arrays:
            return self._arrays[key]
        
        p = self.data_dir / f"{key}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Array {key}.npy not found")
        
        arr = _safe_np_load(p, self.mmap)
        
        # Handle a_1h compatibility (may be T-1 instead of T)
        if key == "a_1h":
            N, T = self.spec.n_stays, self.spec.T
            if arr.shape == (N, T - 1):
                pad = np.zeros((N, T), dtype=arr.dtype)
                pad[:, 1:] = np.asarray(arr)
                arr = pad
        
        self._arrays[key] = arr
        return arr
    
    def has_key(self, key: str) -> bool:
        return (self.data_dir / f"{key}.npy").exists()
    
    def __len__(self):
        return self.spec.n_stays


def load_splits(data_dir: Path, subject_id: np.ndarray) -> Dict[str, np.ndarray]:
    """Load train/val/test splits."""
    npz_path = data_dir / "splits_indices.npz"
    subj_path = data_dir / "splits_subject.json"
    
    if npz_path.exists():
        npz = np.load(str(npz_path), allow_pickle=False)
        
        def pick(*cands):
            for c in cands:
                if c in npz:
                    return c
            return None
        
        k_tr = pick("train", "train_idx", "train_indices")
        k_va = pick("val", "valid", "validation", "val_idx")
        k_te = pick("test", "test_idx", "test_indices")
        
        if k_tr and k_va and k_te:
            return {
                "train": npz[k_tr].astype(np.int64),
                "val": npz[k_va].astype(np.int64),
                "test": npz[k_te].astype(np.int64),
            }
    
    if subj_path.exists():
        js = _read_json(subj_path)
        
        def get_list(d, *cands):
            for c in cands:
                if c in d:
                    return list(d[c])
            return None
        
        tr = get_list(js, "train", "train_subjects")
        va = get_list(js, "val", "valid", "validation", "val_subjects")
        te = get_list(js, "test", "test_subjects")
        
        if tr and va and te:
            tr_set = set(int(x) for x in tr)
            va_set = set(int(x) for x in va)
            te_set = set(int(x) for x in te)
            
            subj = subject_id.astype(np.int64)
            idx = np.arange(len(subj), dtype=np.int64)
            
            return {
                "train": idx[np.isin(subj, list(tr_set))],
                "val": idx[np.isin(subj, list(va_set))],
                "test": idx[np.isin(subj, list(te_set))],
            }
    
    # Fallback: random 80/10/10 split by subject
    LOGGER.warning("No splits found, creating random 80/10/10 split")
    unique_subj = np.unique(subject_id)
    np.random.shuffle(unique_subj)
    n = len(unique_subj)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    
    tr_subj = set(unique_subj[:n_tr].tolist())
    va_subj = set(unique_subj[n_tr:n_tr+n_va].tolist())
    te_subj = set(unique_subj[n_tr+n_va:].tolist())
    
    idx = np.arange(len(subject_id), dtype=np.int64)
    subj = subject_id.astype(np.int64)
    
    return {
        "train": idx[np.isin(subj, list(tr_subj))],
        "val": idx[np.isin(subj, list(va_subj))],
        "test": idx[np.isin(subj, list(te_subj))],
    }


@dataclass
class FeatureScaler:
    """Standardization scaler (train-only fit)."""
    obs_mean: np.ndarray
    obs_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    eps: float = 1e-6
    
    def transform_obs(self, Y: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        mean = self.obs_mean.reshape((1,) * (Y.ndim - 1) + (-1,))
        std = np.maximum(self.obs_std, self.eps).reshape((1,) * (Y.ndim - 1) + (-1,))
        Yz = (Y - mean) / std
        if mask is not None:
            Yz = np.where(mask > 0.5, Yz, 0.0)
        return Yz.astype(np.float32)
    
    def transform_static(self, X: np.ndarray) -> np.ndarray:
        mean = self.static_mean.reshape((1,) * (X.ndim - 1) + (-1,))
        std = np.maximum(self.static_std, self.eps).reshape((1,) * (X.ndim - 1) + (-1,))
        return ((X - mean) / std).astype(np.float32)
    
    def save(self, path: Path):
        np.savez(str(path),
                 obs_mean=self.obs_mean,
                 obs_std=self.obs_std,
                 static_mean=self.static_mean,
                 static_std=self.static_std)
    
    @staticmethod
    def load(path: Path) -> "FeatureScaler":
        npz = np.load(str(path), allow_pickle=False)
        return FeatureScaler(
            obs_mean=npz["obs_mean"].astype(np.float32),
            obs_std=npz["obs_std"].astype(np.float32),
            static_mean=npz["static_mean"].astype(np.float32),
            static_std=npz["static_std"].astype(np.float32),
        )


def fit_scaler(store: ProcessedDataStore, train_idx: np.ndarray, chunk_size: int = 1024) -> FeatureScaler:
    """Fit scaler on train split only (observed values within valid time)."""
    spec = store.spec
    D, S = spec.D, spec.S
    
    # Accumulators for online mean/var
    sum_x = np.zeros((D,), dtype=np.float64)
    sum_x2 = np.zeros((D,), dtype=np.float64)
    cnt = np.zeros((D,), dtype=np.float64)
    
    Y_all = store.get("Y")
    mask_all = store.get("mask")
    time_mask_all = store.get("time_mask")
    X_static_all = store.get("X_static")
    
    for st in range(0, len(train_idx), chunk_size):
        ed = min(st + chunk_size, len(train_idx))
        idx = train_idx[st:ed]
        
        Y = np.asarray(Y_all[idx], dtype=np.float32)
        m = np.asarray(mask_all[idx], dtype=np.float32)
        tm = np.asarray(time_mask_all[idx], dtype=np.float32)
        
        valid = (m > 0.5) & (tm[:, :, None] > 0.5)
        Y_masked = np.where(valid, Y, 0.0).astype(np.float64)
        
        sum_x += Y_masked.sum(axis=(0, 1))
        sum_x2 += (Y_masked ** 2).sum(axis=(0, 1))
        cnt += valid.sum(axis=(0, 1)).astype(np.float64)
    
    cnt_safe = np.maximum(cnt, 1.0)
    obs_mean = (sum_x / cnt_safe).astype(np.float32)
    var = (sum_x2 / cnt_safe) - (obs_mean.astype(np.float64) ** 2)
    obs_std = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)
    
    # Static
    Xtr = np.asarray(X_static_all[train_idx], dtype=np.float32)
    static_mean = Xtr.mean(axis=0).astype(np.float32)
    static_std = np.maximum(Xtr.std(axis=0), 1e-6).astype(np.float32)
    
    return FeatureScaler(obs_mean, obs_std, static_mean, static_std)


def compute_mnar_features(mask: np.ndarray, delta: np.ndarray, time_mask: np.ndarray,
                          window: int = 6) -> np.ndarray:
    """Compute MNAR missingness features: delta, cumcount, missrate, window_freq."""
    T, D = mask.shape
    m = (mask > 0.5).astype(np.float32)
    tm = (time_mask > 0.5).astype(np.float32)
    
    # 1. Delta (time since last obs)
    f1 = delta.astype(np.float32)
    
    # 2. Cumulative obs count
    f2 = np.cumsum(m, axis=0).astype(np.float32)
    
    # 3. Cumulative missing rate
    t_idx = (np.arange(T, dtype=np.float32) + 1.0).reshape(T, 1)
    f3 = (1.0 - f2 / np.maximum(t_idx, 1.0)).astype(np.float32)
    
    # 4. Windowed observation frequency
    c = np.vstack([np.zeros((1, D), dtype=np.float32), np.cumsum(m, axis=0)])
    w = max(1, window)
    left = np.clip(np.arange(T) - (w - 1), 0, T)
    right = np.arange(T) + 1
    f4 = ((c[right] - c[left]) / float(w)).astype(np.float32)
    
    feat = np.concatenate([f1, f2, f3, f4], axis=1)
    return (feat * tm[:, None]).astype(np.float32)


class SepsisDataset(Dataset):
    """Episode-level dataset returning trajectory dicts."""
    
    def __init__(
        self,
        store: ProcessedDataStore,
        indices: np.ndarray,
        scaler: Optional[FeatureScaler] = None,
        normalize: bool = True,
        compute_mnar: bool = True,
        mnar_window: int = 6,
    ):
        super().__init__()
        self.store = store
        self.indices = indices.astype(np.int64)
        self.scaler = scaler
        self.normalize = normalize and scaler is not None
        self.compute_mnar = compute_mnar
        self.mnar_window = mnar_window
        self.spec = store.spec
        # Cache time grids once (robust to either (T,) or (N,T))
        self._t_1h = np.asarray(self.store.get("t_1h"), dtype=np.int32)
        self._t_4h = np.asarray(self.store.get("t_4h"), dtype=np.int32)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        
        # Load arrays
        Y = np.asarray(self.store.get("Y")[i], dtype=np.float32).copy()
        mask = np.asarray(self.store.get("mask")[i], dtype=np.float32).copy()
        delta = np.asarray(self.store.get("delta")[i], dtype=np.float32).copy()
        time_mask = np.asarray(self.store.get("time_mask")[i], dtype=np.float32)
        X_static = np.asarray(self.store.get("X_static")[i], dtype=np.float32).copy()
        
        a_4h = np.asarray(self.store.get("a_4h")[i], dtype=np.int64).copy()
        r_4h = np.asarray(self.store.get("r_4h")[i], dtype=np.float32).copy()
        done_4h = np.asarray(self.store.get("done_4h")[i], dtype=np.float32).copy()
        step_mask_4h = np.asarray(self.store.get("step_mask_4h")[i], dtype=np.float32)
        
        e_rad = np.asarray(self.store.get("e_rad")[i], dtype=np.float32).copy()
        e_micro = np.asarray(self.store.get("e_micro")[i], dtype=np.float32).copy()
        m_text = np.asarray(self.store.get("m_text")[i], dtype=np.float32).copy()
        
        # Robust time grid fetch
        t_1h_np = self._t_1h[i] if self._t_1h.ndim == 2 else self._t_1h
        t_4h_np = self._t_4h[i] if self._t_4h.ndim == 2 else self._t_4h

        t_1h = np.asarray(t_1h_np, dtype=np.int32)
        t_4h = np.asarray(t_4h_np, dtype=np.int32)
        
        # MNAR features
        if self.compute_mnar:
            mnar_feat = compute_mnar_features(mask, delta, time_mask, self.mnar_window)
        else:
            mnar_feat = np.zeros((self.spec.T, self.spec.D * 4), dtype=np.float32)
        
        # Normalize
        if self.normalize:
            Y = self.scaler.transform_obs(Y, mask)
            X_static = self.scaler.transform_static(X_static)
        
        # Zero-out invalid regions
        invalid_t = (time_mask <= 0.5)
        if np.any(invalid_t):
            Y[invalid_t, :] = 0.0
            mask[invalid_t, :] = 0.0
            delta[invalid_t, :] = 0.0
            mnar_feat[invalid_t, :] = 0.0
        
        invalid_k = (step_mask_4h <= 0.5)
        if np.any(invalid_k):
            a_4h[invalid_k] = 0
            r_4h[invalid_k] = 0.0
            done_4h[invalid_k] = 0.0
            e_rad[invalid_k, :] = 0.0
            e_micro[invalid_k, :] = 0.0
            m_text[invalid_k, ...] = 0.0
        
        # Derived info
        done_idx = int(np.argmax(done_4h)) if done_4h.sum() > 0 else (self.spec.K - 1)
        terminal_reward = float(r_4h[done_idx])
        y_mortality = 1.0 if terminal_reward < 0 else 0.0
        
        return {
            "idx": torch.tensor(i, dtype=torch.long),
            "Y": torch.from_numpy(Y),
            "mask": torch.from_numpy(mask),
            "delta": torch.from_numpy(delta),
            "time_mask": torch.from_numpy(time_mask),
            "mnar_feat": torch.from_numpy(mnar_feat),
            "X_static": torch.from_numpy(X_static),
            "t_1h": torch.from_numpy(t_1h),
            "t_4h": torch.from_numpy(t_4h),
            "a_4h": torch.from_numpy(a_4h),
            "r_4h": torch.from_numpy(r_4h),
            "done_4h": torch.from_numpy(done_4h),
            "step_mask_4h": torch.from_numpy(step_mask_4h),
            "e_rad": torch.from_numpy(e_rad),
            "e_micro": torch.from_numpy(e_micro),
            "m_text": torch.from_numpy(m_text),
            "done_index": torch.tensor(done_idx, dtype=torch.long),
            "y_mortality": torch.tensor(y_mortality, dtype=torch.float32),
            
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack batch items into batched tensors."""
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def make_dataloaders(cfg: DataConfig, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureScaler, DataSpec]:
    """Create train/val/test dataloaders."""
    data_dir = _as_path(cfg.data_dir)
    store = ProcessedDataStore(data_dir, mmap=cfg.mmap)
    spec = store.spec

    subject_id = store.get("subject_id")
    splits = load_splits(data_dir, subject_id)

    LOGGER.info(f"Data: N={spec.n_stays}, T={spec.T}, K={spec.K}, D={spec.D}, S={spec.S}")
    LOGGER.info(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Scaler
    scaler_path = data_dir / "scaler.npz"
    if scaler_path.exists():
        scaler = FeatureScaler.load(scaler_path)
        LOGGER.info(f"Loaded scaler from {scaler_path}")
    else:
        scaler = fit_scaler(store, splits["train"])
        scaler.save(scaler_path)
        LOGGER.info(f"Fitted and saved scaler to {scaler_path}")

    # Datasets
    ds_train = SepsisDataset(store, splits["train"], scaler, cfg.normalize_obs, cfg.compute_mnar_features, cfg.mnar_window_hours)
    ds_val = SepsisDataset(store, splits["val"], scaler, cfg.normalize_obs, cfg.compute_mnar_features, cfg.mnar_window_hours)
    ds_test = SepsisDataset(store, splits["test"], scaler, cfg.normalize_obs, cfg.compute_mnar_features, cfg.mnar_window_hours)

    # Generator (controls shuffling deterministically)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        generator=g,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader, test_loader, scaler, spec


# =============================================================================
# Part 3: Neural Networks
# =============================================================================

def _assert_shape(t: torch.Tensor, shape: Tuple[Optional[int], ...], name: str):
    """Runtime shape check."""
    if t.dim() != len(shape):
        raise ValueError(f"{name} dim mismatch: expected {len(shape)}, got {t.dim()}")
    for i, s in enumerate(shape):
        if s is not None and t.shape[i] != s:
            raise ValueError(f"{name} shape[{i}] mismatch: expected {s}, got {t.shape[i]}")


class MLP(nn.Module):
    """Simple MLP with LayerNorm and GELU."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class SinusoidalTimeEnc(nn.Module):
    """Sinusoidal positional encoding for time."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float32)
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class GRUDCell(nn.Module):
    """GRU-D cell with missingness-aware decay + explicit MNAR feature fusion."""

    def __init__(
        self,
        d_obs: int,
        d_hidden: int,
        d_mnar: int,
        enable_mnar_fusion: bool = True,
        d_mnar_embed: int = 32,
        d_delta_embed: int = 16,
        d_time_embed: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_obs = d_obs
        self.d_hidden = d_hidden
        self.d_mnar = d_mnar
        self.enable_mnar_fusion = enable_mnar_fusion
        self.d_mnar_embed = d_mnar_embed
        self.d_delta_embed = d_delta_embed
        self.d_time_embed = d_time_embed

        # Learned feature mean for decay imputation
        self.x_mean = nn.Parameter(torch.zeros(d_obs))

        # Decay parameters (depend on delta)
        self.lin_gamma_x = nn.Linear(d_obs, d_obs)
        self.lin_gamma_h = nn.Linear(d_obs, d_hidden)

        # Delta embedding (depends on delta)
        self.delta_embed = nn.Sequential(
            nn.Linear(d_obs, d_delta_embed),
            nn.LayerNorm(d_delta_embed),
            nn.GELU(),
        )

        # Time embedding
        self.time_enc = SinusoidalTimeEnc(d_time_embed if d_time_embed % 2 == 0 else d_time_embed + 1)
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_enc.dim, d_time_embed),
            nn.LayerNorm(d_time_embed),
            nn.GELU(),
        )

        # MNAR embedding
        if self.enable_mnar_fusion:
            self.mnar_embed = nn.Sequential(
                nn.Linear(d_mnar, d_mnar_embed),
                nn.LayerNorm(d_mnar_embed),
                nn.GELU(),
            )
        else:
            self.mnar_embed = None

        # GRU cell input dimension
        # x_hat (D) + mask (D) + delta_emb + time_emb + (optional) mnar_emb
        in_dim = d_obs + d_obs + d_delta_embed + d_time_embed
        if self.enable_mnar_fusion:
            in_dim += d_mnar_embed

        self.gru = nn.GRUCell(in_dim, d_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_t, m_t, d_t, mnar_t, h_prev, valid_t, t_val):
        """
        Args:
            x_t: (B, D)
            m_t: (B, D)
            d_t: (B, D)
            mnar_t: (B, d_mnar)
            h_prev: (B, H)
            valid_t: (B,)
            t_val: (B,)
        Returns:
            h_next: (B, H)
        """
        # Sanitize inputs
        x_t = torch.nan_to_num(x_t.float(), 0.0)
        m_t = torch.nan_to_num(m_t.float(), 0.0).clamp(0, 1)
        d_t = torch.nan_to_num(d_t.float(), 0.0).clamp_min(0)

        if mnar_t is None:
            mnar_t = torch.zeros(x_t.shape[0], self.d_mnar, device=x_t.device, dtype=torch.float32)
        else:
            mnar_t = torch.nan_to_num(mnar_t.float(), 0.0)

        # Compute decays
        gamma_x = torch.exp(-F.relu(self.lin_gamma_x(d_t)))
        gamma_h = torch.exp(-F.relu(self.lin_gamma_h(d_t)))

        # Hidden decay
        h = gamma_h * h_prev

        # Input decay-to-mean imputation
        x_hat = m_t * x_t + (1 - m_t) * (gamma_x * x_t + (1 - gamma_x) * self.x_mean)

        # Embeddings
        d_emb = self.delta_embed(d_t)                  # (B, d_delta_embed)
        t_emb = self.time_proj(self.time_enc(t_val))   # (B, d_time_embed)

        parts = [x_hat, m_t, d_emb, t_emb]

        # MNAR embedding
        if self.enable_mnar_fusion and self.mnar_embed is not None:
            mnar_emb = self.mnar_embed(mnar_t)         # (B, d_mnar_embed)
            parts.append(mnar_emb)

        inp = self.drop(torch.cat(parts, dim=-1))
        h_new = self.gru(inp, h)

        # Keep previous if invalid
        valid_mask = (valid_t > 0.5).unsqueeze(-1)
        return torch.where(valid_mask, h_new, h_prev)

class GRUDEncoder(nn.Module):
    """GRU-D encoder over 1h observation grid with explicit MNAR feature fusion."""

    def __init__(
        self,
        d_obs: int,
        d_hidden: int,
        d_mnar: int,
        enable_mnar_fusion: bool = True,
        d_mnar_embed: int = 32,
        d_delta_embed: int = 16,
        d_time_embed: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_obs = d_obs
        self.d_mnar = d_mnar

        self.cell = GRUDCell(
            d_obs=d_obs,
            d_hidden=d_hidden,
            d_mnar=d_mnar,
            enable_mnar_fusion=enable_mnar_fusion,
            d_mnar_embed=d_mnar_embed,
            d_delta_embed=d_delta_embed,
            d_time_embed=d_time_embed,
            dropout=dropout,
        )

    def forward(self, Y, mask, delta, mnar_feat, time_mask, t_1h, h0=None):
        """
        Args:
            Y: (B, T, D)
            mask: (B, T, D)
            delta: (B, T, D)
            mnar_feat: (B, T, d_mnar)
            time_mask: (B, T)
            t_1h: (B, T) or (T,)
            h0: optional (B, H)
        Returns:
            h_seq: (B, T, H)
            h_last: (B, H)
        """
        B, T, D = Y.shape
        device = Y.device

        if t_1h.dim() == 1:
            t_seq = t_1h.unsqueeze(0).expand(B, T)
        else:
            t_seq = t_1h

        if mnar_feat is None:
            mnar_feat = torch.zeros(B, T, self.d_mnar, device=device, dtype=torch.float32)

        h = torch.zeros(B, self.d_hidden, device=device) if h0 is None else h0
        h_seq = []

        for t in range(T):
            h = self.cell(
                Y[:, t],
                mask[:, t],
                delta[:, t],
                mnar_feat[:, t],
                h,
                time_mask[:, t],
                t_seq[:, t],
            )
            h_seq.append(h)

        h_seq = torch.stack(h_seq, dim=1)  # (B, T, H)

        # Get last valid hidden
        lengths = time_mask.sum(dim=1).clamp_min(1).long()
        idx = (lengths - 1).view(B, 1, 1).expand(-1, 1, self.d_hidden)
        h_last = h_seq.gather(1, idx).squeeze(1)

        return h_seq, h_last

class DecisionStateProjector(nn.Module):
    """Project 1h hidden states to 4h decision states."""
    
    def __init__(self, d_hidden: int, d_state: int, mode: str = "at_decision"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Sequential(
            nn.Linear(d_hidden, d_state),
            nn.LayerNorm(d_state),
            nn.GELU(),
        )
        
        # Index buffer
        self.register_buffer("_idx_map", torch.empty(0, dtype=torch.long))
    
    def _compute_idx_map(self, t_1h, t_4h):
        """Map 4h decision times to 1h indices."""
        t1 = t_1h[0] if t_1h.dim() == 2 else t_1h
        t4 = t_4h[0] if t_4h.dim() == 2 else t_4h
        
        idx = []
        for v in t4.tolist():
            matches = (t1 == int(v)).nonzero(as_tuple=False)
            if matches.numel() > 0:
                idx.append(int(matches[0].item()))
            else:
                dif = (t1 - int(v)).abs()
                idx.append(int(dif.argmin().item()))
        return torch.tensor(idx, dtype=torch.long, device=t1.device)
    
    def forward(self, h_seq, t_1h, t_4h, step_mask_4h):
        """
        Args:
            h_seq: (B, T, H)
            t_1h: (B, T) or (T,)
            t_4h: (B, K) or (K,)
            step_mask_4h: (B, K)
        Returns:
            states: (B, K, d_state)
        """
        B, T, H = h_seq.shape
        K = t_4h.shape[-1]
        
        # Compute index map (cached)
        if self._idx_map.numel() != K:
            self._idx_map = self._compute_idx_map(t_1h, t_4h)
        
        idx_map = self._idx_map.to(h_seq.device)
        
        if self.mode == "at_decision":
            # Select hidden at decision times
            idx = idx_map.view(1, K, 1).expand(B, -1, H)
            h_k = h_seq.gather(1, idx)  # (B, K, H)
        else:
            # Block pool (average over 4h blocks)
            h_k_list = []
            for k in range(K):
                start = idx_map[k].item()
                end = min(start + 4, T)
                h_block = h_seq[:, start:end, :].mean(dim=1)
                h_k_list.append(h_block)
            h_k = torch.stack(h_k_list, dim=1)
        
        states = self.proj(h_k)
        
        # Zero out invalid steps
        states = states * step_mask_4h.unsqueeze(-1)
        
        return states


class TextFusion(nn.Module):
    """Cross-attention text fusion for radiology + microbiology, with meta-text gating."""

    def __init__(self, d_state: int, d_text: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_state = d_state
        self.d_text = d_text

        # Project text to state dimension
        self.text_proj = nn.Linear(d_text, d_state)

        # Cross-attention: query=state, key/value=text tokens
        self.cross_attn = nn.MultiheadAttention(d_state, n_heads, dropout=dropout, batch_first=True)

        # Meta projection: we reduce m_text to a scalar per step, then map 1 -> d_state
        self.meta_proj = nn.Sequential(
            nn.Linear(1, d_state),
            nn.LayerNorm(d_state),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Adaptive gate
        # concat: [states, h_text, states*h_text, meta_emb] => 4*d_state
        self.gate = nn.Sequential(
            nn.Linear(d_state * 4, d_state),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, states, e_rad, e_micro, m_text, step_mask_4h):
        """
        Args:
            states: (B, K, d_state)
            e_rad: (B, K, d_text)
            e_micro: (B, K, d_text)
            m_text: (B, K, M) or (B, K) or (B, K, 1)
            step_mask_4h: (B, K)
        Returns:
            fused_states: (B, K, d_state)
        """
        B, K, _ = states.shape

        # Project text embeddings
        e_rad = torch.nan_to_num(e_rad.float(), 0.0)
        e_micro = torch.nan_to_num(e_micro.float(), 0.0)
        rad_proj = self.text_proj(e_rad)      # (B, K, d_state)
        micro_proj = self.text_proj(e_micro)  # (B, K, d_state)

        # Normalize m_text shape => (B, K, M)
        if m_text is None:
            m_text_in = torch.zeros((B, K, 1), device=states.device, dtype=torch.float32)
        else:
            m_text = torch.nan_to_num(m_text.float(), 0.0)
            if m_text.dim() == 2:
                m_text_in = m_text.unsqueeze(-1)  # (B,K,1)
            elif m_text.dim() == 3:
                m_text_in = m_text                # (B,K,M)
            else:
                raise ValueError(f"m_text has unsupported dim: {m_text.dim()}")

        # Reduce meta to scalar per step: (B,K,1)
        if m_text_in.shape[-1] != 1:
            m_scalar = m_text_in.mean(dim=-1, keepdim=True)
        else:
            m_scalar = m_text_in

        # Meta embedding: (B,K,d_state)
        meta_emb = self.meta_proj(m_scalar)

        # Per-step attention
        q = states.reshape(B * K, 1, -1)  # (B*K, 1, d_state)
        kv = torch.stack([rad_proj, micro_proj], dim=2).reshape(B * K, 2, -1)  # (B*K, 2, d_state)

        h_text, _ = self.cross_attn(q, kv, kv)     # (B*K, 1, d_state)
        h_text = h_text.reshape(B, K, -1)          # (B, K, d_state)

        # Gate
        gate_input = torch.cat([states, h_text, states * h_text, meta_emb], dim=-1)  # (B,K,4d)
        g = self.gate(gate_input)

        fused = g * h_text + (1 - g) * states
        fused = self.norm(states + self.drop(fused - states))

        # Zero out invalid steps
        fused = fused * step_mask_4h.unsqueeze(-1)

        return fused


class ActorHead(nn.Module):
    """Policy head outputting action logits."""
    
    def __init__(self, d_state: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(d_state, hidden_dim, n_actions, n_layers=2, dropout=0.1)
    
    def forward(self, states):
        """Returns logits (B, K, n_actions)."""
        return self.net(states)


class QHead(nn.Module):
    """Q-value head outputting Q(s,a) for all actions."""
    
    def __init__(self, d_state: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(d_state, hidden_dim, n_actions, n_layers=2, dropout=0.1)
    
    def forward(self, states):
        """Returns Q values for all actions (B, K, n_actions)."""
        return self.net(states)


class ValueHead(nn.Module):
    """State value head V(s)."""
    
    def __init__(self, d_state: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(d_state, hidden_dim, 1, n_layers=2, dropout=0.1)
    
    def forward(self, states):
        """Returns V(s) (B, K, 1) -> squeeze to (B, K)."""
        return self.net(states).squeeze(-1)


class OutcomeHead(nn.Module):
    """Mortality prediction head."""
    
    def __init__(self, d_state: int, hidden_dim: int = 64):
        super().__init__()
        self.net = MLP(d_state, hidden_dim, 1, n_layers=2, dropout=0.1)
    
    def forward(self, states):
        """Predict mortality logit from last valid state."""
        return self.net(states)


class ReconstructionHead(nn.Module):
    """Observation + mask + text reconstruction head for state verification."""
    
    def __init__(self, d_state: int, d_obs: int, d_text: int = 0, hidden_dim: int = 128):
        super().__init__()
        self.d_text = d_text
        
        # Observation reconstruction (MSE target)
        self.obs_net = MLP(d_state, hidden_dim, d_obs, n_layers=2, dropout=0.1)
        
        # Mask reconstruction (BCE target) - predicts which variables were observed
        self.mask_net = MLP(d_state, hidden_dim, d_obs, n_layers=2, dropout=0.1)
        
        # Text reconstruction (MSE target) - only if text fusion is enabled
        self.text_net = None
        if d_text > 0:
            self.text_net = MLP(d_state, hidden_dim, d_text, n_layers=2, dropout=0.1)
    
    def forward(self, states):
        """
        Args:
            states: (B, K, d_state)
        Returns:
            obs_pred: (B, K, d_obs)
            mask_pred: (B, K, d_obs) logits
            text_pred: (B, K, d_text) or None
        """
        obs_pred = self.obs_net(states)
        mask_pred = self.mask_net(states)
        text_pred = self.text_net(states) if self.text_net is not None else None
        return obs_pred, mask_pred, text_pred


class SepsisIQLModel(nn.Module):
    """Complete model for Sepsis IQL."""
    
    def __init__(self, spec: DataSpec, enc_cfg: EncoderConfig, aux_cfg: AuxLossConfig):
        super().__init__()
        self.spec = spec
        self.enc_cfg = enc_cfg
        self.aux_cfg = aux_cfg

        D = spec.D
        S = spec.S
        d_hidden = enc_cfg.d_hidden_1h
        d_state = enc_cfg.d_state
        text_dim = spec.text_dim
        n_actions = spec.n_actions

        # MNAR feature dim at 1h grid
        d_mnar = D * 4  # delta, cumcount, missrate, windowfreq

        # GRU-D encoder (now accepts mnar_feat)
        self.encoder = GRUDEncoder(
            d_obs=D,
            d_hidden=d_hidden,
            d_mnar=d_mnar,
            enable_mnar_fusion=enc_cfg.enable_mnar_fusion,
            d_mnar_embed=enc_cfg.d_mnar_embed,
            d_delta_embed=enc_cfg.d_delta_embed,
            d_time_embed=enc_cfg.d_time_embed,
            dropout=enc_cfg.gru_dropout,
        )

        # Static embedding
        self.static_embed = nn.Sequential(
            nn.Linear(S, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
        )

        # Decision state projector (combines GRU-D hidden + static)
        self.state_proj = DecisionStateProjector(
            d_hidden=d_hidden + d_hidden // 2,
            d_state=d_state,
            mode=enc_cfg.state_mode,
        )

        # Text fusion (optional)
        self.text_fusion = None
        if enc_cfg.enable_text_fusion and text_dim > 0:
            self.text_fusion = TextFusion(
                d_state=d_state,
                d_text=text_dim,
                n_heads=enc_cfg.text_n_heads,
                dropout=enc_cfg.text_dropout,
            )

        # RL heads
        self.actor = ActorHead(d_state, n_actions)
        self.q1 = QHead(d_state, n_actions)
        self.q2 = QHead(d_state, n_actions)
        self.value = ValueHead(d_state)

        # Auxiliary heads
        self.outcome_head = None
        if aux_cfg.enable_outcome:
            self.outcome_head = OutcomeHead(d_state)

        self.recon_head = None
        if aux_cfg.enable_reconstruction:
            self.recon_head = ReconstructionHead(
                d_state=d_state,
                d_obs=D,
                d_text=text_dim if enc_cfg.enable_text_fusion else 0,
            )
    
    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode batch to decision states.

        Returns:
            states: (B, K, d_state)
        """
        Y = batch["Y"]
        mask = batch["mask"]
        delta = batch["delta"]
        mnar_feat = batch.get("mnar_feat", None)
        time_mask = batch["time_mask"]
        X_static = batch["X_static"]
        t_1h = batch["t_1h"]
        t_4h = batch["t_4h"]
        step_mask = batch["step_mask_4h"]

        B = Y.shape[0]

        # GRU-D encoding (now uses mnar_feat explicitly)
        h_seq, h_last = self.encoder(Y, mask, delta, mnar_feat, time_mask, t_1h)

        # Static embedding (broadcast to all timesteps)
        static_emb = self.static_embed(X_static)  # (B, d_hidden//2)
        static_exp = static_emb.unsqueeze(1).expand(-1, h_seq.shape[1], -1)

        # Combine
        h_combined = torch.cat([h_seq, static_exp], dim=-1)  # (B, T, d_hidden + d_hidden//2)

        # Project to decision states
        states = self.state_proj(h_combined, t_1h, t_4h, step_mask)  # (B, K, d_state)

        # Text fusion
        if self.text_fusion is not None:
            e_rad = batch["e_rad"]
            e_micro = batch["e_micro"]
            m_text = batch["m_text"]
            states = self.text_fusion(states, e_rad, e_micro, m_text, step_mask)

        return states
    
    def forward_rl(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward through RL heads."""
        return {
            "actor_logits": self.actor(states),
            "q1": self.q1(states),
            "q2": self.q2(states),
            "value": self.value(states),
        }
    
    def forward_aux(
        self,
        states: torch.Tensor,
        step_mask_4h: Optional[torch.Tensor] = None,
        done_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward through auxiliary heads.

        Args:
            states: (B, K, d_state)
            step_mask_4h: optional (B, K) for masked pooling
            done_index: optional (B,) terminal index for each trajectory

        Returns:
            dict of aux outputs
        """
        out: Dict[str, torch.Tensor] = {}

        if self.outcome_head is not None:
            B, K, d = states.shape

            if done_index is not None:
                idx = done_index.view(B, 1, 1).expand(-1, 1, d)
                pooled = states.gather(1, idx).squeeze(1)  # (B, d_state)
            elif step_mask_4h is not None:
                w = step_mask_4h.unsqueeze(-1)  # (B, K, 1)
                denom = w.sum(dim=1).clamp_min(1.0)
                pooled = (states * w).sum(dim=1) / denom  # (B, d_state)
            else:
                pooled = states[:, -1, :]  # fallback

            out["outcome_logits"] = self.outcome_head(pooled).squeeze(-1)  # (B,)

        if self.recon_head is not None:
            obs_pred, mask_pred = self.recon_head(states)
            out["obs_pred"] = obs_pred
            out["mask_pred"] = mask_pred

        return out

# =============================================================================
# Part 4: Pure IQL Agent
# =============================================================================

class PureIQLAgent:
    """
    Pure IQL implementation with correct 3-loss formulation.
    
    Core losses (NO alpha/entropy/BC/KL by default):
    1. Q-TD: L_Q = E[(Q(s,a) - y)²] where y = r + γ(1-d)V_target(s')
    2. V-expectile: L_V = E[|τ - 1(δ<0)| · δ²] where δ = Q_min(s,a) - V(s)
    3. π-AWBC: L_π = -E[clip(exp(A/β), w_max) · log π(a|s)]
    
    Optional stabilizers (for ablation):
    - Entropy stabilizer (only if entropy drops too low)
    """
    
    def __init__(
        self,
        model: SepsisIQLModel,
        iql_cfg: IQLConfig,
        train_cfg: TrainConfig,
        aux_cfg: AuxLossConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.iql_cfg = iql_cfg
        self.train_cfg = train_cfg
        self.aux_cfg = aux_cfg
        self.device = device

        # Target V (always)
        self.target_value = copy.deepcopy(model.value).to(device)
        for p in self.target_value.parameters():
            p.requires_grad = False

        # Optional Target Q (stabilizer)
        self.target_q1 = None
        self.target_q2 = None
        if self.iql_cfg.use_target_q:
            self.target_q1 = copy.deepcopy(model.q1).to(device)
            self.target_q2 = copy.deepcopy(model.q2).to(device)
            for p in self.target_q1.parameters():
                p.requires_grad = False
            for p in self.target_q2.parameters():
                p.requires_grad = False

        # === Parameter groups with explicit names ===
        self.q1_params = list(model.q1.parameters())
        self.q2_params = list(model.q2.parameters())
        self.value_params = list(model.value.parameters())
        self.actor_params = list(model.actor.parameters())

        self.encoder_params = list(model.encoder.parameters()) + \
                            list(model.static_embed.parameters()) + \
                            list(model.state_proj.parameters())
        if model.text_fusion is not None:
            self.encoder_params += list(model.text_fusion.parameters())

        self.aux_params = []
        if model.outcome_head is not None:
            self.aux_params += list(model.outcome_head.parameters())
        if model.recon_head is not None:
            self.aux_params += list(model.recon_head.parameters())

        # Optimizers with NAMED param groups for rollback
        self.opt_rl = torch.optim.AdamW([
            {"params": self.q1_params, "lr": train_cfg.lr_critic, "name": "q1"},
            {"params": self.q2_params, "lr": train_cfg.lr_critic, "name": "q2"},
            {"params": self.value_params, "lr": train_cfg.lr_value, "name": "value"},
            {"params": self.actor_params, "lr": train_cfg.lr_actor, "name": "actor"},
        ], weight_decay=train_cfg.weight_decay)

        self.opt_encoder = torch.optim.AdamW(
            [{"params": self.encoder_params, "lr": train_cfg.lr_encoder, "name": "encoder"}],
            weight_decay=train_cfg.weight_decay,
        )

        if self.aux_params:
            self.opt_aux = torch.optim.AdamW(
                [{"params": self.aux_params, "lr": train_cfg.lr_aux, "name": "aux"}],
                weight_decay=train_cfg.weight_decay,
            )
        else:
            self.opt_aux = None

        # Training state
        self.step = 0
        self.collapse_counter = 0
        self.best_metrics = {"fqe_lcb": -float("inf")}

        # Metrics buffer
        self.metrics_buffer = []

        
    def _set_encoder_mode(self, mode: str):
        if mode == "train":
            self.model.encoder.train()
            self.model.static_embed.train()
            self.model.state_proj.train()
            if self.model.text_fusion is not None:
                self.model.text_fusion.train()
        elif mode == "eval":
            self.model.encoder.eval()
            self.model.static_embed.eval()
            self.model.state_proj.eval()
            if self.model.text_fusion is not None:
                self.model.text_fusion.eval()
    
    def _set_rl_heads_mode(self, mode: str):
        """Set RL heads to train/eval mode."""
        if mode == "train":
            self.model.actor.train()
            self.model.q1.train()
            self.model.q2.train()
            self.model.value.train()
        elif mode == "eval":
            self.model.actor.eval()
            self.model.q1.eval()
            self.model.q2.eval()
            self.model.value.eval()
    
    def update_targets(self):
        """Polyak averaging for target networks (V always; Q optional)."""
        tau = self.iql_cfg.tau

        # V target
        for p, tp in zip(self.model.value.parameters(), self.target_value.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

        # Optional Q targets
        if self.iql_cfg.use_target_q and self.target_q1 is not None and self.target_q2 is not None:
            for p, tp in zip(self.model.q1.parameters(), self.target_q1.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
            for p, tp in zip(self.model.q2.parameters(), self.target_q2.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
    
    def _get_training_phase(self) -> str:
        """Determine current training phase."""
        if self.step < self.train_cfg.encoder_pretrain_steps:
            return "pretrain_encoder"
        elif self.step < self.train_cfg.encoder_freeze_steps:
            return "freeze_encoder"
        else:
            return "finetune"
    
    def _should_update_actor(self) -> bool:
        """Check if actor should be updated this step."""
        if self.step < self.train_cfg.critic_warmup_steps:
            return False
        return (self.step - self.train_cfg.critic_warmup_steps) % self.train_cfg.actor_update_every == 0
    
    def compute_q_loss(self, states, actions, rewards, dones, step_mask, states_next):
        """
        Q-learning loss with TD target using V (not max Q).
        
        y = r + γ(1-d) V_target(s')
        L_Q = E[(Q(s,a) - y)² * mask]
        """
        B, K = states.shape[:2]
        
        # Compute TD target using target V
        with torch.no_grad():
            # Shift states: s' for step k is s at step k+1
            # For last step, use zero (terminal)
            v_next = self.target_value(states_next)  # (B, K)
            
            # TD target
            td_target = rewards + self.iql_cfg.gamma * (1 - dones) * v_next
            
            # Soft clip target (not hard clip Q values)
            td_target = td_target.clamp(-self.iql_cfg.td_target_clip, self.iql_cfg.td_target_clip)
        
        # Q predictions for taken actions
        q1_all = self.model.q1(states)  # (B, K, n_actions)
        q2_all = self.model.q2(states)
        
        actions_exp = actions.unsqueeze(-1)  # (B, K, 1)
        q1 = q1_all.gather(-1, actions_exp).squeeze(-1)  # (B, K)
        q2 = q2_all.gather(-1, actions_exp).squeeze(-1)
        
        # Loss
        if self.iql_cfg.use_huber_loss:
            loss_q1 = F.huber_loss(q1, td_target, reduction="none", delta=self.iql_cfg.huber_delta)
            loss_q2 = F.huber_loss(q2, td_target, reduction="none", delta=self.iql_cfg.huber_delta)
        else:
            loss_q1 = (q1 - td_target).pow(2)
            loss_q2 = (q2 - td_target).pow(2)
        
        # Masked mean
        mask_sum = step_mask.sum().clamp_min(1.0)
        loss_q = ((loss_q1 + loss_q2) * step_mask).sum() / mask_sum
        
        return loss_q, q1.detach(), q2.detach(), td_target.detach()
    
    def compute_v_loss(self, states, actions, step_mask):
        """
        Value loss via expectile regression.

        δ = Q_min(s,a) - V(s)
        L_V = E[|τ - 1(δ<0)| · δ² * mask]

        If use_target_q: compute Q_min using target_q networks for stability.
        """
        with torch.no_grad():
            q1_net = self.target_q1 if (self.iql_cfg.use_target_q and self.target_q1 is not None) else self.model.q1
            q2_net = self.target_q2 if (self.iql_cfg.use_target_q and self.target_q2 is not None) else self.model.q2

            q1_all = q1_net(states)
            q2_all = q2_net(states)

            actions_exp = actions.unsqueeze(-1)
            q1 = q1_all.gather(-1, actions_exp).squeeze(-1)
            q2 = q2_all.gather(-1, actions_exp).squeeze(-1)
            q_min = torch.min(q1, q2)

        v = self.model.value(states)

        diff = q_min - v
        tau = self.iql_cfg.expectile
        weight = torch.where(diff > 0, tau, 1.0 - tau)

        loss_v = (weight * diff.pow(2) * step_mask).sum() / step_mask.sum().clamp_min(1.0)

        return loss_v, v.detach()
    
    def compute_actor_loss(self, states, actions, step_mask):
        """
        Actor loss via Advantage-Weighted BC (AWBC).

        A(s,a) = Q_min(s,a) - V(s)
        w = clip(exp(A/β), w_max)
        L_π = -E[w · log π(a|s) * mask]

        If use_target_q: compute Q_min using target_q networks for stability.
        """
        with torch.no_grad():
            q1_net = self.target_q1 if (self.iql_cfg.use_target_q and self.target_q1 is not None) else self.model.q1
            q2_net = self.target_q2 if (self.iql_cfg.use_target_q and self.target_q2 is not None) else self.model.q2

            q1_all = q1_net(states)
            q2_all = q2_net(states)

            actions_exp = actions.unsqueeze(-1)
            q1 = q1_all.gather(-1, actions_exp).squeeze(-1)
            q2 = q2_all.gather(-1, actions_exp).squeeze(-1)
            q_min = torch.min(q1, q2)

            v = self.model.value(states)
            adv = q_min - v

            # Advantage normalization (batch-wise, masked)
            if self.iql_cfg.adv_normalize:
                valid_adv = adv[step_mask > 0.5]
                if valid_adv.numel() > 1:
                    adv_mean = valid_adv.mean()
                    adv_std = valid_adv.std().clamp_min(1e-4)
                    adv = (adv - adv_mean) / adv_std

            # AWBC weights
            awbc_weight = torch.exp(adv / self.iql_cfg.awbc_beta)
            awbc_weight = awbc_weight.clamp(max=self.iql_cfg.awbc_clip)

        logits = self.model.actor(states)
        logits = logits.clamp(-self.iql_cfg.logit_clip, self.iql_cfg.logit_clip)

        log_probs = F.log_softmax(logits, dim=-1)
        log_pi_a = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        loss_actor = -(awbc_weight * log_pi_a * step_mask).sum() / step_mask.sum().clamp_min(1.0)

        # Entropy monitoring
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            probs_safe = probs.clamp(min=self.iql_cfg.log_prob_floor)
            entropy_per_step = -(probs_safe * probs_safe.log()).sum(dim=-1)
            avg_entropy = (entropy_per_step * step_mask).sum() / step_mask.sum().clamp_min(1.0)
            entropy = avg_entropy.item()

        # Optional entropy stabilizer
        if self.iql_cfg.use_entropy_stabilizer and entropy < self.iql_cfg.entropy_stabilizer_min:
            probs_for_ent = F.softmax(logits, dim=-1)
            probs_for_ent = probs_for_ent.clamp(min=self.iql_cfg.log_prob_floor)
            entropy_loss_term = -(probs_for_ent * probs_for_ent.log()).sum(dim=-1)
            avg_entropy_loss = (entropy_loss_term * step_mask).sum() / step_mask.sum().clamp_min(1.0)
            loss_actor = loss_actor - self.iql_cfg.entropy_stabilizer_coef * avg_entropy_loss

        return loss_actor, entropy

    
    def compute_aux_losses(self, batch, states):
        """
        Compute auxiliary losses (reconstruction, outcome).
        
        Reconstruction loss (Eq. 6 in paper):
            L_recon = λ_obs ||ŷ_h - y_h||² + λ_mask BCE(m̂_h, m_h) + λ_text ||ê_h - e_h||²
        
        The mask reconstruction ensures the state captures MNAR patterns.
        """
        losses = {}
        total = torch.tensor(0.0, device=self.device)
        
        step_mask = batch["step_mask_4h"]
        B, K = step_mask.shape
        
        # Outcome prediction (mortality) - this works correctly
        if self.aux_cfg.enable_outcome and self.model.outcome_head is not None:
            # Predict mortality from last valid state
            done_idx = batch["done_index"]  # (B,)
            
            # Gather last valid state
            idx = done_idx.view(B, 1, 1).expand(-1, 1, states.shape[-1])
            last_state = states.gather(1, idx).squeeze(1)  # (B, d_state)
            
            outcome_logits = self.model.outcome_head(last_state).squeeze(-1)  # (B,)
            y_mortality = batch["y_mortality"]
            
            loss_outcome = F.binary_cross_entropy_with_logits(outcome_logits, y_mortality)
            losses["outcome"] = loss_outcome.item()
            total = total + self.aux_cfg.w_outcome * loss_outcome
        
        # Reconstruction loss
        if self.aux_cfg.enable_reconstruction and self.model.recon_head is not None:
            # === Aggregate 1h observations to 4h decision points ===
            Y = batch["Y"]              # (B, T, D)
            mask = batch["mask"]        # (B, T, D)
            t_1h = batch["t_1h"]        # (B, T) or (T,)
            t_4h = batch["t_4h"]        # (B, K) or (K,)
            
            T, D = Y.shape[1], Y.shape[2]
            
            # Build index map from 4h to 1h (use last observation in each 4h window)
            t1 = t_1h[0] if t_1h.dim() == 2 else t_1h  # (T,)
            t4 = t_4h[0] if t_4h.dim() == 2 else t_4h  # (K,)
            
            # For each 4h step k at time t4[k], find the corresponding 1h index
            # We use the observation at exactly that time point (end of window)
            idx_map = []
            for k, t_val in enumerate(t4.tolist()):
                matches = (t1 == int(t_val)).nonzero(as_tuple=False)
                if matches.numel() > 0:
                    idx_map.append(int(matches[0].item()))
                else:
                    # Fallback: find closest
                    dif = (t1 - int(t_val)).abs()
                    idx_map.append(int(dif.argmin().item()))
            
            idx_map = torch.tensor(idx_map, dtype=torch.long, device=self.device)  # (K,)
            
            # Gather targets at decision points
            idx_exp = idx_map.view(1, K, 1).expand(B, -1, D)
            Y_target = Y.gather(1, idx_exp)       # (B, K, D)
            mask_target = mask.gather(1, idx_exp)  # (B, K, D)
            
            # Forward through reconstruction head
            obs_pred, mask_pred, text_pred = self.model.recon_head(states)
            
            # === Observation reconstruction (MSE, only on observed values) ===
            # Reconstruct only where mask_target == 1
            obs_diff = (obs_pred - Y_target) ** 2  # (B, K, D)
            obs_loss_per_elem = obs_diff * mask_target  # zero out unobserved
            
            # Normalize by number of observed elements
            n_observed = (mask_target * step_mask.unsqueeze(-1)).sum().clamp_min(1.0)
            loss_obs = (obs_loss_per_elem * step_mask.unsqueeze(-1)).sum() / n_observed
            
            # === Mask reconstruction (BCE) ===
            # Predict which variables were observed at each decision step
            mask_loss = F.binary_cross_entropy_with_logits(
                mask_pred, mask_target, reduction="none"
            )  # (B, K, D)
            
            n_mask_elems = (step_mask.unsqueeze(-1).expand(-1, -1, D)).sum().clamp_min(1.0)
            loss_mask = (mask_loss * step_mask.unsqueeze(-1)).sum() / n_mask_elems
            
            # === Text reconstruction (MSE, if text fusion enabled) ===
            loss_text = torch.tensor(0.0, device=self.device)
            if text_pred is not None:
                e_rad = batch["e_rad"]      # (B, K, d_text)
                e_micro = batch["e_micro"]  # (B, K, d_text)
                m_text = batch["m_text"]    # (B, K, M) or (B, K)
                
                # Combine text embeddings (average of available modalities)
                if m_text.dim() == 2:
                    m_text = m_text.unsqueeze(-1)  # (B, K, 1)
                
                # m_text might be (B, K, 2) for [rad_avail, micro_avail]
                if m_text.shape[-1] >= 2:
                    m_rad = m_text[..., 0:1]    # (B, K, 1)
                    m_micro = m_text[..., 1:2]  # (B, K, 1)
                else:
                    # Single indicator - assume both modalities share it
                    m_rad = m_text
                    m_micro = m_text
                
                # Weighted average of available text embeddings
                text_sum = m_rad * e_rad + m_micro * e_micro  # (B, K, d_text)
                text_count = (m_rad + m_micro).clamp_min(1e-6)  # avoid div by zero
                text_target = text_sum / text_count  # (B, K, d_text)
                
                # Text availability mask (at least one modality present)
                text_avail = ((m_rad + m_micro) > 0.5).float().squeeze(-1)  # (B, K)
                
                # MSE loss on text reconstruction
                text_diff = (text_pred - text_target) ** 2  # (B, K, d_text)
                text_mask = (step_mask * text_avail).unsqueeze(-1)  # (B, K, 1)
                
                n_text = text_mask.sum().clamp_min(1.0) * text_pred.shape[-1]
                loss_text = (text_diff * text_mask).sum() / n_text
            
            # Combine reconstruction losses
            loss_recon = (
                self.aux_cfg.w_recon_obs * loss_obs +
                self.aux_cfg.w_recon_mask * loss_mask +
                self.aux_cfg.w_recon_text * loss_text
            )
            
            losses["recon_obs"] = loss_obs.item()
            losses["recon_mask"] = loss_mask.item()
            losses["recon_text"] = loss_text.item() if isinstance(loss_text, torch.Tensor) else loss_text
            losses["recon"] = loss_recon.item()
            
            total = total + loss_recon
        
        return total, losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Training phases:
        1. pretrain_encoder: Only aux losses, encoder learning (encoder=train, RL=no grad)
        2. freeze_encoder: Freeze encoder (encoder=eval, no grad), train RL heads only
        3. finetune: Allow encoder gradient from RL loss with scaled LR
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        phase = self._get_training_phase()
        metrics = {"step": self.step, "phase": phase}
        
        # Extract tensors
        actions = batch["a_4h"]
        rewards = batch["r_4h"]
        dones = batch["done_4h"]
        step_mask = batch["step_mask_4h"]
        
        # === Phase 1: Pretrain encoder with aux losses ===
        if phase == "pretrain_encoder":
            # Encoder in train mode, RL heads don't matter
            self._set_encoder_mode("train")
            
            self.opt_encoder.zero_grad()
            if self.opt_aux is not None:
                self.opt_aux.zero_grad()
            
            states = self.model.encode(batch)
            loss_aux, aux_metrics = self.compute_aux_losses(batch, states)
            
            if isinstance(loss_aux, torch.Tensor) and loss_aux.item() > 0:
                loss_aux.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder_params + self.aux_params, self.train_cfg.grad_clip)
                self.opt_encoder.step()
                if self.opt_aux is not None:
                    self.opt_aux.step()
            
            metrics["loss_aux"] = loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux
            metrics.update(aux_metrics)
        
        # === Phase 2: Freeze encoder, train RL heads ===
        elif phase == "freeze_encoder":
            self._set_encoder_mode("eval")
            self._set_rl_heads_mode("train")
            
            # Ensure encoder params don't receive gradients
            for p in self.encoder_params:
                p.requires_grad = False
            
            # Forward pass (no gradient through encoder)
            with torch.no_grad():
                states = self.model.encode(batch)
            states = states.detach()  # Extra safety: detach to ensure no grad flows back
            
            # Compute next states (shifted by 1 step)
            B, K, d = states.shape
            states_next = torch.zeros_like(states)
            states_next[:, :-1, :] = states[:, 1:, :]
            
            # Q loss
            loss_q, q1_pred, q2_pred, td_target = self.compute_q_loss(
                states, actions, rewards, dones, step_mask, states_next
            )
            
            # V loss
            loss_v, v_pred = self.compute_v_loss(states, actions, step_mask)
            
            # Actor loss (delayed)
            loss_actor = torch.tensor(0.0, device=self.device)
            entropy = 0.0
            if self._should_update_actor():
                loss_actor, entropy = self.compute_actor_loss(states, actions, step_mask)
            
            # Backward for RL only
            self.opt_rl.zero_grad()
            loss_rl = loss_q + loss_v + loss_actor
            loss_rl.backward()
            torch.nn.utils.clip_grad_norm_(
                self.q1_params + self.q2_params + self.value_params + self.actor_params,
                self.train_cfg.grad_clip
            )
            self.opt_rl.step()
            
            # Update targets
            self.update_targets()
            
            # Record metrics
            metrics["loss_q"] = loss_q.item()
            metrics["loss_v"] = loss_v.item()
            metrics["loss_actor"] = loss_actor.item() if isinstance(loss_actor, torch.Tensor) else loss_actor
            metrics["loss_aux"] = 0.0
            metrics["q_mean"] = ((q1_pred + q2_pred) / 2 * step_mask).sum().item() / step_mask.sum().item()
            metrics["v_mean"] = (v_pred * step_mask).sum().item() / step_mask.sum().item()
            metrics["entropy"] = entropy
        
        # === Phase 3: Finetune - encoder receives RL gradients ===
        else:  # finetune
            # Both encoder and RL heads in train mode
            self._set_encoder_mode("train")
            self._set_rl_heads_mode("train")
            
            # Re-enable encoder gradients
            for p in self.encoder_params:
                p.requires_grad = True
            
            # Forward pass WITH gradient through encoder
            states = self.model.encode(batch)
            
            # Compute next states
            B, K, d = states.shape
            states_next = torch.zeros_like(states)
            states_next[:, :-1, :] = states[:, 1:, :].detach()  # detach next states to prevent double backprop
            
            # Q loss
            loss_q, q1_pred, q2_pred, td_target = self.compute_q_loss(
                states, actions, rewards, dones, step_mask, states_next
            )
            
            # V loss
            loss_v, v_pred = self.compute_v_loss(states, actions, step_mask)
            
            # Actor loss (delayed)
            loss_actor = torch.tensor(0.0, device=self.device)
            entropy = 0.0
            if self._should_update_actor():
                loss_actor, entropy = self.compute_actor_loss(states, actions, step_mask)
            
            # Aux losses
            loss_aux = torch.tensor(0.0, device=self.device)
            aux_metrics = {}
            if self.opt_aux is not None:
                loss_aux, aux_metrics = self.compute_aux_losses(batch, states.detach())
            
            # Backward: RL loss updates BOTH RL heads AND encoder
            self.opt_rl.zero_grad()
            self.opt_encoder.zero_grad()
            
            loss_rl = loss_q + loss_v + loss_actor
            loss_rl.backward(retain_graph=(isinstance(loss_aux, torch.Tensor) and loss_aux.item() > 0))
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.q1_params + self.q2_params + self.value_params + self.actor_params,
                self.train_cfg.grad_clip
            )
            torch.nn.utils.clip_grad_norm_(self.encoder_params, self.train_cfg.grad_clip)
            
            # Step RL optimizer
            self.opt_rl.step()
            
            # Step encoder optimizer with scaled LR
            # Temporarily scale down encoder LR for finetuning
            for pg in self.opt_encoder.param_groups:
                pg["lr"] = self.train_cfg.lr_encoder * self.train_cfg.encoder_finetune_lr_scale
            self.opt_encoder.step()
            
            # Aux loss (separate backward if needed)
            if self.opt_aux is not None and isinstance(loss_aux, torch.Tensor) and loss_aux.item() > 0:
                self.opt_aux.zero_grad()
                loss_aux.backward()
                torch.nn.utils.clip_grad_norm_(self.aux_params, self.train_cfg.grad_clip)
                self.opt_aux.step()
            
            # Update targets
            self.update_targets()
            
            # Record metrics
            metrics["loss_q"] = loss_q.item()
            metrics["loss_v"] = loss_v.item()
            metrics["loss_actor"] = loss_actor.item() if isinstance(loss_actor, torch.Tensor) else loss_actor
            metrics["loss_aux"] = loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux
            metrics["q_mean"] = ((q1_pred + q2_pred) / 2 * step_mask).sum().item() / step_mask.sum().item()
            metrics["v_mean"] = (v_pred * step_mask).sum().item() / step_mask.sum().item()
            metrics["entropy"] = entropy
            metrics.update(aux_metrics)
        
        self.step += 1
        self.metrics_buffer.append(metrics)
        
        return metrics
    
    def compute_collapse_metrics(self, loader: DataLoader, max_batches: int = 10) -> Dict[str, float]:
        """Compute policy collapse detection metrics."""
        self.model.eval()
        
        all_probs = []
        all_masks = []
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                states = self.model.encode(batch)
                logits = self.model.actor(states)
                probs = F.softmax(logits, dim=-1)
                
                all_probs.append(probs.cpu())
                all_masks.append(batch["step_mask_4h"].cpu())
        
        all_probs = torch.cat(all_probs, dim=0)  # (N, K, A)
        all_masks = torch.cat(all_masks, dim=0)  # (N, K)
        
        # Flatten valid steps
        valid = all_masks > 0.5
        probs_valid = all_probs[valid]  # (M, A)
        
        if probs_valid.numel() == 0:
            return {"entropy": 0.0, "top1_prob": 1.0, "unique_actions": 1}
        
        # Entropy
        entropy = -(probs_valid * (probs_valid + 1e-8).log()).sum(dim=-1).mean().item()
        
        # Top-1 probability
        top1_prob = probs_valid.max(dim=-1)[0].mean().item()
        
        # Number of unique predicted actions
        actions = probs_valid.argmax(dim=-1)
        unique_actions = len(torch.unique(actions))
        
        return {
            "entropy": entropy,
            "top1_prob": top1_prob,
            "unique_actions": unique_actions,
        }
    
    def check_collapse_and_rollback(self, metrics: Dict[str, float], checkpoint_dir: Path) -> bool:
        """
        Check for policy collapse and trigger rollback if needed.
        
        Returns True if rollback was triggered.
        """
        is_collapsed = (
            metrics["entropy"] < self.train_cfg.collapse_entropy_threshold or
            metrics["top1_prob"] > self.train_cfg.collapse_top1_threshold
        )
        
        if is_collapsed:
            self.collapse_counter += 1
            LOGGER.warning(f"Collapse detected ({self.collapse_counter}/{self.train_cfg.collapse_patience}): "
                          f"entropy={metrics['entropy']:.3f} nats, top1={metrics['top1_prob']:.3f}")
            
            if self.collapse_counter >= self.train_cfg.collapse_patience:
                # Trigger rollback
                healthy_ckpt = checkpoint_dir / "last_healthy.pt"
                if healthy_ckpt.exists():
                    LOGGER.warning(f"Rolling back to {healthy_ckpt}")
                    self.load(healthy_ckpt)
                    
                    # Adjust hyperparameters
                    old_beta = self.iql_cfg.awbc_beta
                    old_clip = self.iql_cfg.awbc_clip
                    self.iql_cfg.awbc_beta *= self.train_cfg.rollback_beta_scale
                    self.iql_cfg.awbc_clip *= self.train_cfg.rollback_clip_scale
                    
                    # Find and adjust actor LR by name
                    actor_lr_adjusted = False
                    for pg in self.opt_rl.param_groups:
                        if pg.get("name") == "actor":
                            old_lr = pg["lr"]
                            pg["lr"] *= self.train_cfg.rollback_lr_scale
                            LOGGER.warning(f"  Actor LR: {old_lr:.2e} -> {pg['lr']:.2e}")
                            actor_lr_adjusted = True
                            break
                    
                    if not actor_lr_adjusted:
                        LOGGER.warning("  WARNING: Could not find 'actor' param group to adjust LR!")
                    
                    LOGGER.warning(f"  beta: {old_beta:.2f} -> {self.iql_cfg.awbc_beta:.2f}")
                    LOGGER.warning(f"  clip: {old_clip:.1f} -> {self.iql_cfg.awbc_clip:.1f}")
                else:
                    LOGGER.warning(f"No healthy checkpoint found at {healthy_ckpt}, cannot rollback")
                
                self.collapse_counter = 0
                return True
        else:
            self.collapse_counter = 0
        
        return False

    def save(self, path: Path):
        """Save checkpoint."""
        state = {
            "model": self.model.state_dict(),
            "target_value": self.target_value.state_dict(),
            "target_q1": self.target_q1.state_dict() if self.target_q1 is not None else None,
            "target_q2": self.target_q2.state_dict() if self.target_q2 is not None else None,
            "opt_rl": self.opt_rl.state_dict(),
            "opt_encoder": self.opt_encoder.state_dict(),
            "opt_aux": self.opt_aux.state_dict() if self.opt_aux else None,
            "step": self.step,
            "iql_cfg": asdict(self.iql_cfg),
            "best_metrics": self.best_metrics,
        }
        torch.save(state, path)
    
    def load(self, path: Path):
        """Load checkpoint."""
        state = torch.load(path, map_location=self.device)
        
        # FIX: Remove _idx_map from state_dict before loading
        # It will be recomputed on first forward pass
        model_state = state["model"]
        keys_to_remove = [k for k in model_state.keys() if "_idx_map" in k]
        for k in keys_to_remove:
            del model_state[k]
        
        self.model.load_state_dict(model_state, strict=False)
        
        self.target_value.load_state_dict(state["target_value"])

        # Restore optional target Q if enabled
        if self.iql_cfg.use_target_q:
            if self.target_q1 is None or self.target_q2 is None:
                self.target_q1 = copy.deepcopy(self.model.q1).to(self.device)
                self.target_q2 = copy.deepcopy(self.model.q2).to(self.device)
                for p in self.target_q1.parameters():
                    p.requires_grad = False
                for p in self.target_q2.parameters():
                    p.requires_grad = False

            if state.get("target_q1") is not None and state.get("target_q2") is not None:
                self.target_q1.load_state_dict(state["target_q1"])
                self.target_q2.load_state_dict(state["target_q2"])
            else:
                self.target_q1.load_state_dict(self.model.q1.state_dict())
                self.target_q2.load_state_dict(self.model.q2.state_dict())

        self.opt_rl.load_state_dict(state["opt_rl"])
        self.opt_encoder.load_state_dict(state["opt_encoder"])
        if self.opt_aux and state["opt_aux"]:
            self.opt_aux.load_state_dict(state["opt_aux"])

        self.step = state["step"]
        self.best_metrics = state.get("best_metrics", {"fqe_lcb": -float("inf")})

        if "iql_cfg" in state:
            saved_cfg = state["iql_cfg"]
            self.iql_cfg.awbc_beta = saved_cfg.get("awbc_beta", self.iql_cfg.awbc_beta)
            self.iql_cfg.awbc_clip = saved_cfg.get("awbc_clip", self.iql_cfg.awbc_clip)
            self.iql_cfg.use_target_q = saved_cfg.get("use_target_q", self.iql_cfg.use_target_q)
    
    def get_policy_probs(self, states: torch.Tensor, apply_softening: bool = False, soften_eps: float = 0.01) -> torch.Tensor:
        """
        Get policy probabilities.
        
        Args:
            states: (B, K, d_state) encoded states
            apply_softening: if True, apply AI Clinician-style 99%/1% softening
            soften_eps: epsilon for softening (default 0.01 = 99%/1%)
        
        Returns:
            probs: (B, K, n_actions) policy probabilities
        
        NOTE: For consistency, both WIS and FQE should use the same policy.
        If using softening for WIS, also use it for FQE.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model.actor(states)
            logits = logits.clamp(-self.iql_cfg.logit_clip, self.iql_cfg.logit_clip)
            probs = F.softmax(logits, dim=-1)
            
            if apply_softening:
                # AI Clinician-style softening: greedy action gets (1-eps), rest get eps/(A-1)
                A = probs.shape[-1]
                greedy = probs.argmax(dim=-1, keepdim=True)
                softened = torch.full_like(probs, soften_eps / (A - 1))
                softened.scatter_(-1, greedy, 1.0 - soften_eps)
                probs = softened
        
        return probs


# =============================================================================
# Part 5: Behavior Policy & OPE
# =============================================================================

class BehaviorPolicyEstimator:
    """
    Behavior policy estimator for OPE.
    
    Key insight: Must use FROZEN encoder representation to avoid distribution shift.
    """
    
    def __init__(self, d_state: int, n_actions: int, device: torch.device):
        self.d_state = d_state
        self.n_actions = n_actions
        self.device = device
        
        self.bc_head = nn.Sequential(
            nn.Linear(d_state, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_actions),
        ).to(device)
        
        self.ope_encoder = None  # Will be set during OPE
        self.is_fitted = False
    
    def prepare_for_ope(self, model: SepsisIQLModel, train_loader: DataLoader, ope_cfg: OPEConfig):
        """
        Prepare behavior policy for OPE.
        
        1. Copy and freeze encoder
        2. Train BC head on frozen encoder representations
        """
        LOGGER.info("Preparing behavior policy for OPE...")
        
        # 1. Copy and freeze encoder
        self.ope_encoder = copy.deepcopy(model).to(self.device)
        for p in self.ope_encoder.parameters():
            p.requires_grad = False
        self.ope_encoder.eval()
        
        # 2. Train BC head
        optimizer = torch.optim.Adam(self.bc_head.parameters(), lr=ope_cfg.bc_lr)
        
        for epoch in range(ope_cfg.bc_epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with torch.no_grad():
                    states = self.ope_encoder.encode(batch)
                
                actions = batch["a_4h"]
                step_mask = batch["step_mask_4h"]
                
                logits = self.bc_head(states)
                
                # Flatten for cross-entropy
                B, K, A = logits.shape
                logits_flat = logits.view(-1, A)
                actions_flat = actions.view(-1)
                mask_flat = step_mask.view(-1)
                
                # Masked cross-entropy
                loss = F.cross_entropy(logits_flat, actions_flat, reduction="none")
                loss = (loss * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            LOGGER.info(f"BC epoch {epoch+1}/{ope_cfg.bc_epochs}: loss={total_loss/n_batches:.4f}")
        
        self.is_fitted = True
        
        # Freeze BC head
        for p in self.bc_head.parameters():
            p.requires_grad = False
        self.bc_head.eval()
    
    def get_probs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get behavior policy probabilities."""
        if not self.is_fitted:
            raise RuntimeError("BehaviorPolicyEstimator not fitted. Call prepare_for_ope first.")
        
        with torch.no_grad():
            states = self.ope_encoder.encode(batch)
            logits = self.bc_head(states)
            probs = F.softmax(logits, dim=-1)
        
        return probs


def soften_policy(probs: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """
    Apply AI Clinician-style policy softening.
    
    Greedy action gets (1-eps), rest get eps/(A-1).
    
    NOTE: This function is kept for backward compatibility.
    Prefer using agent.get_policy_probs(states, apply_softening=True) for consistency.
    """
    A = probs.shape[-1]
    greedy = probs.argmax(dim=-1, keepdim=True)
    
    softened = torch.full_like(probs, eps / (A - 1))
    softened.scatter_(-1, greedy, 1.0 - eps)
    
    return softened


class WISEstimator:
    """
    Weighted Importance Sampling estimator (AI Clinician aligned).
    
    Features:
    - Policy softening (99%/1%)
    - Ratio truncation
    - Bootstrap confidence intervals
    - ESS reporting
    """
    
    def __init__(self, gamma: float = 0.99, ope_cfg: OPEConfig = None):
        self.gamma = gamma
        self.cfg = ope_cfg or OPEConfig()
    
    def compute_trajectory_wis(
        self,
        pi_eval: torch.Tensor,
        pi_behavior: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-trajectory WIS estimates.

        NOTE:
        - pi_eval is assumed to be the policy you want to evaluate (softened or not),
            so we do NOT soften again here.
        - terminal index is determined primarily by dones, not by step_mask length.
        """
        B, K, A = pi_eval.shape

        actions_exp = actions.unsqueeze(-1)
        pi_e_a = pi_eval.gather(-1, actions_exp).squeeze(-1)      # (B,K)
        pi_b_a = pi_behavior.gather(-1, actions_exp).squeeze(-1)  # (B,K)

        rho = pi_e_a / (pi_b_a + 1e-8)
        rho = rho.clamp(max=self.cfg.ratio_clip)
        rho = rho * step_mask + (1.0 - step_mask)  # invalid steps => ratio 1

        rho_cumul = torch.cumprod(rho, dim=1)

        # terminal index: first done==1 among valid steps; else last valid step
        done_valid = (dones > 0.5) & (step_mask > 0.5)  # (B,K)
        has_done = done_valid.any(dim=1)  # (B,)

        # first done index (if multiple, pick earliest)
        first_done = done_valid.float().argmax(dim=1)  # (B,) (works because False->0)
        last_valid = (step_mask.sum(dim=1).long() - 1).clamp_min(0)

        term_idx = torch.where(has_done, first_done, last_valid)  # (B,)
        rho_final = rho_cumul.gather(1, term_idx.unsqueeze(1)).squeeze(1)  # (B,)

        discount = torch.pow(self.gamma, torch.arange(K, device=rewards.device).float())
        discounted_rewards = rewards * discount.unsqueeze(0) * step_mask
        returns = discounted_rewards.sum(dim=1)

        wis_values = rho_final * returns
        return wis_values, rho_final

    def estimate_with_bootstrap(
        self,
        eval_agent: PureIQLAgent,
        behavior_policy: BehaviorPolicyEstimator,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Compute WIS estimate with bootstrap CI.
        """
        eval_agent.model.eval()

        all_wis = []
        all_rho = []

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Encode states
                states = eval_agent.model.encode(batch)

                # Evaluation policy probabilities (softened)
                pi_eval = eval_agent.get_policy_probs(
                    states,
                    apply_softening=True,
                    soften_eps=self.cfg.soften_eps,
                )

                # Behavior policy probabilities
                pi_behavior = behavior_policy.get_probs(batch)

                actions = batch["a_4h"]
                rewards = batch["r_4h"]
                dones = batch["done_4h"]
                step_mask = batch["step_mask_4h"]

                wis_vals, rho_final = self.compute_trajectory_wis(
                    pi_eval, pi_behavior, actions, rewards, dones, step_mask
                )

                all_wis.append(wis_vals.cpu())
                all_rho.append(rho_final.cpu())

        all_wis = torch.cat(all_wis, dim=0).numpy()
        all_rho = torch.cat(all_rho, dim=0).numpy()

        # Self-normalized WIS: sum(rho*G)/sum(rho)
        wis_mean = float(np.mean(all_wis) / (np.mean(all_rho) + 1e-8))

        # Bootstrap CI
        n = len(all_wis)
        bootstrap_estimates = []
        for _ in range(self.cfg.bootstrap_n):
            idx = np.random.choice(n, n, replace=True)
            sample_wis = all_wis[idx]
            sample_rho = all_rho[idx]
            est = float(np.mean(sample_wis) / (np.mean(sample_rho) + 1e-8))
            bootstrap_estimates.append(est)

        bootstrap_estimates = np.array(bootstrap_estimates, dtype=np.float64)
        alpha = self.cfg.bootstrap_alpha
        lcb = float(np.percentile(bootstrap_estimates, 100.0 * alpha / 2.0))
        ucb = float(np.percentile(bootstrap_estimates, 100.0 * (1.0 - alpha / 2.0)))

        # Effective Sample Size
        ess = float((np.sum(all_rho) ** 2) / (np.sum(all_rho ** 2) + 1e-8))

        return {
            "wis_mean": wis_mean,
            "wis_lcb": lcb,
            "wis_ucb": ucb,
            "wis_std": float(np.std(bootstrap_estimates)),
            "ess": ess,
            "max_ratio": float(np.max(all_rho)),
            "mean_ratio": float(np.mean(all_rho)),
        }

class FQEEstimator:
    """
    Fitted Q-Evaluation estimator.
    """
    
    def __init__(self, d_state: int, n_actions: int, gamma: float = 0.99, device: torch.device = None):
        self.d_state = d_state
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device or torch.device("cpu")
        
        # FQE Q-network
        self.fqe_q = nn.Sequential(
            nn.Linear(d_state, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, n_actions),
        ).to(self.device)
        
        self.fqe_q_target = copy.deepcopy(self.fqe_q)
        for p in self.fqe_q_target.parameters():
            p.requires_grad = False
    
    def fit(
        self,
        eval_agent: PureIQLAgent,
        train_loader: DataLoader,
        ope_cfg: OPEConfig,
    ):
        """
        Fit FQE Q-function.
        """
        LOGGER.info("Fitting FQE...")
        
        optimizer = torch.optim.Adam(self.fqe_q.parameters(), lr=ope_cfg.fqe_lr)
        
        eval_agent.model.eval()
        
        step = 0
        train_iter = iter(train_loader)
        
        while step < ope_cfg.fqe_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():
                states = eval_agent.model.encode(batch)
                # Use softened policy for consistency!
                pi_eval = eval_agent.get_policy_probs(
                    states,
                    apply_softening=True,
                    soften_eps=ope_cfg.soften_eps
                )
            
            actions = batch["a_4h"]
            rewards = batch["r_4h"]
            dones = batch["done_4h"]
            step_mask = batch["step_mask_4h"]
            
            B, K, _ = states.shape
            
            # Next states
            states_next = torch.zeros_like(states)
            states_next[:, :-1, :] = states[:, 1:, :]
            
            # TD target: r + γ(1-d) E_π[Q(s',a')]
            with torch.no_grad():
                q_next = self.fqe_q_target(states_next)
                v_next = (pi_eval * q_next).sum(dim=-1)  # Using softened policy
                td_target = rewards + self.gamma * (1 - dones) * v_next
            
            # Q prediction
            q_pred = self.fqe_q(states)
            q_a = q_pred.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            
            # Loss
            loss = ((q_a - td_target).pow(2) * step_mask).sum() / step_mask.sum().clamp_min(1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Target update
            tau = ope_cfg.fqe_target_tau
            for p, tp in zip(self.fqe_q.parameters(), self.fqe_q_target.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
            
            step += 1
            
            if step % 2000 == 0:
                LOGGER.info(f"FQE step {step}/{ope_cfg.fqe_steps}: loss={loss.item():.4f}")
    
    def estimate(
        self,
        eval_agent: PureIQLAgent,
        data_loader: DataLoader,
        bootstrap_n: int = 2000,
        apply_softening: bool = True,
        soften_eps: float = 0.01,
    ) -> Dict[str, float]:
        """
        Estimate policy value using fitted Q-function.
        
        Args:
            eval_agent: trained IQL agent
            data_loader: data to evaluate on
            bootstrap_n: number of bootstrap samples
            apply_softening: whether to apply policy softening (should match WIS!)
            soften_eps: softening epsilon
        """
        eval_agent.model.eval()
        self.fqe_q.eval()
        
        all_values = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                states = eval_agent.model.encode(batch)
                
                # Use same policy as WIS for consistency!
                pi_eval = eval_agent.get_policy_probs(
                    states,
                    apply_softening=apply_softening,
                    soften_eps=soften_eps
                )
                
                q = self.fqe_q(states)
                v = (pi_eval * q).sum(dim=-1)  # (B, K)
                
                # Initial value (step 0)
                v0 = v[:, 0]
                all_values.append(v0.cpu())
        
        all_values = torch.cat(all_values, dim=0).numpy()
        
        # Bootstrap CI
        n = len(all_values)
        bootstrap_estimates = []
        
        for _ in range(bootstrap_n):
            idx = np.random.choice(n, n, replace=True)
            bootstrap_estimates.append(np.mean(all_values[idx]))
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        lcb = np.percentile(bootstrap_estimates, 2.5)
        ucb = np.percentile(bootstrap_estimates, 97.5)
        
        return {
            "fqe_mean": float(np.mean(all_values)),
            "fqe_lcb": float(lcb),
            "fqe_ucb": float(ucb),
            "fqe_std": float(np.std(bootstrap_estimates)),
        }

def compute_dose_gap_analysis(
    eval_agent: PureIQLAgent,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Compute dose gap vs mortality analysis (AI Clinician Fig.3d/e style).
    """
    eval_agent.model.eval()
    
    all_gaps = []
    all_mortality = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            states = eval_agent.model.encode(batch)
            probs = eval_agent.get_policy_probs(states)
            
            # Clinician actions
            a_clinician = batch["a_4h"]  # (B, K)
            
            # Policy actions (greedy)
            a_policy = probs.argmax(dim=-1)  # (B, K)
            
            step_mask = batch["step_mask_4h"]
            mortality = batch["y_mortality"]
            
            # Compute gap per valid step
            # Action = fluids_level * 3 + vaso_level
            # Gap = |fluids_diff| + |vaso_diff|
            fluids_clin = a_clinician // 3
            vaso_clin = a_clinician % 3
            fluids_pol = a_policy // 3
            vaso_pol = a_policy % 3
            
            gap = (fluids_clin - fluids_pol).abs() + (vaso_clin - vaso_pol).abs()  # (B, K)
            
            # Average gap per trajectory (over valid steps)
            avg_gap = (gap.float() * step_mask).sum(dim=1) / step_mask.sum(dim=1).clamp_min(1.0)
            
            all_gaps.append(avg_gap.cpu())
            all_mortality.append(mortality.cpu())
    
    all_gaps = torch.cat(all_gaps, dim=0).numpy()
    all_mortality = torch.cat(all_mortality, dim=0).numpy()
    
    # Binned analysis
    results = {
        "overall_agreement": float(np.mean(all_gaps == 0)),
        "mean_gap": float(np.mean(all_gaps)),
    }
    
    # Mortality by gap bins
    for g in range(5):
        mask = (all_gaps >= g) & (all_gaps < g + 1)
        if mask.sum() > 0:
            results[f"gap_{g}_mortality"] = float(np.mean(all_mortality[mask]))
            results[f"gap_{g}_count"] = int(mask.sum())
    
    # Agreement vs disagreement
    agree_mask = all_gaps < 0.5
    disagree_mask = ~agree_mask
    
    if agree_mask.sum() > 0:
        results["agree_mortality"] = float(np.mean(all_mortality[agree_mask]))
        results["agree_count"] = int(agree_mask.sum())
    
    if disagree_mask.sum() > 0:
        results["disagree_mortality"] = float(np.mean(all_mortality[disagree_mask]))
        results["disagree_count"] = int(disagree_mask.sum())
    
    return results


# =============================================================================
# Part 6: Training Loop
# =============================================================================

def train(cfg: ExperimentConfig):
    """Main training loop."""
    # Setup
    set_seed(cfg.train.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Device: {device}")

    # Create run directory
    run_dir = Path(cfg.run_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Run directory: {run_dir}")

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Data
    LOGGER.info("Loading data...")
    train_loader, val_loader, test_loader, scaler, spec = make_dataloaders(cfg.data, cfg.train.seed)
    LOGGER.info(f"Spec: T={spec.T}, K={spec.K}, D={spec.D}, A={spec.n_actions}, text_dim={spec.text_dim}")

    # Model
    LOGGER.info("Creating model...")
    model = SepsisIQLModel(spec, cfg.encoder, cfg.aux)

    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model parameters: {n_params:,}")

    # Agent
    agent = PureIQLAgent(model, cfg.iql, cfg.train, cfg.aux, device)

    # Training loop
    LOGGER.info("Starting training...")
    train_iter = iter(train_loader)

    best_fqe_lcb = -float("inf")

    # === Early stopping state ===
    no_improve = 0  # counts consecutive FQE-fast evals without sufficient LCB improvement

    start_time = time.time()

    while agent.step < cfg.train.total_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Train step
        metrics = agent.train_step(batch)

        # Logging
        if agent.step % cfg.train.log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = agent.step / elapsed if elapsed > 0 else 0

            phase = metrics.get("phase", "unk")
            log_str = f"Step {agent.step}/{cfg.train.total_steps} ({steps_per_sec:.1f} it/s)"
            log_str += f" | phase={phase}"

            if phase == "pretrain_encoder":
                log_str += f" | Aux={metrics.get('loss_aux', 0.0):.4f}"
                if "outcome" in metrics:
                    log_str += f" | outcome_bce={metrics.get('outcome', 0.0):.4f}"
            else:
                log_str += f" | Q={metrics.get('loss_q', 0.0):.4f}"
                log_str += f" | V={metrics.get('loss_v', 0.0):.4f}"
                log_str += f" | π={metrics.get('loss_actor', 0.0):.4f}"
                if "entropy" in metrics and metrics["entropy"] > 0:
                    log_str += f" | H={metrics['entropy']:.3f}"

            LOGGER.info(log_str)

        # Evaluation
        if agent.step % cfg.train.eval_every == 0 and agent.step > 0:
            LOGGER.info("=" * 60)
            LOGGER.info(f"Evaluation at step {agent.step}")

            # Collapse metrics (cheap)
            collapse_metrics = agent.compute_collapse_metrics(val_loader)
            LOGGER.info(
                f"Collapse check: entropy={collapse_metrics['entropy']:.3f}, "
                f"top1={collapse_metrics['top1_prob']:.3f}, "
                f"unique={collapse_metrics['unique_actions']}"
            )

            # Check for collapse and rollback
            rolled_back = agent.check_collapse_and_rollback(collapse_metrics, run_dir)
            if rolled_back:
                continue

            # Save healthy checkpoint
            agent.save(run_dir / "last_healthy.pt")

            # Run FQE only occasionally (expensive)
            if agent.step % cfg.train.fqe_eval_every == 0:
                LOGGER.info("Running FQE (training-time quick eval)...")
                fqe = FQEEstimator(cfg.encoder.d_state, spec.n_actions, cfg.iql.gamma, device)

                # Use fewer FQE steps during training to reduce cost
                ope_cfg_fast = copy.deepcopy(cfg.ope)
                ope_cfg_fast.fqe_steps = int(cfg.train.fqe_steps_during_training)

                fqe.fit(agent, train_loader, ope_cfg_fast)
                fqe_results = fqe.estimate(
                    agent,
                    val_loader,
                    bootstrap_n=cfg.ope.bootstrap_n,
                    apply_softening=True,
                    soften_eps=cfg.ope.soften_eps,
                )

                LOGGER.info(
                    f"FQE-fast: mean={fqe_results['fqe_mean']:.4f} "
                    f"[{fqe_results['fqe_lcb']:.4f}, {fqe_results['fqe_ucb']:.4f}]"
                )

                # === Model selection + early stopping based on FQE LCB ===
                lcb = float(fqe_results["fqe_lcb"])
                delta = float(getattr(cfg.train, "early_stop_delta", 0.0))

                improved = (lcb > best_fqe_lcb + delta)
                if improved:
                    best_fqe_lcb = lcb
                    no_improve = 0
                    agent.save(run_dir / "best_model.pt")
                    LOGGER.info(f"New best model! FQE LCB = {best_fqe_lcb:.4f}")
                else:
                    no_improve += 1
                    LOGGER.info(
                        f"No improvement on FQE LCB. "
                        f"patience={no_improve}/{cfg.train.early_stop_patience} "
                        f"(best_lcb={best_fqe_lcb:.4f}, current_lcb={lcb:.4f}, delta={delta:.1e})"
                    )

                # Trigger early stop only after min steps
                if getattr(cfg.train, "enable_early_stop", False) and agent.step >= cfg.train.early_stop_min_steps:
                    if no_improve >= cfg.train.early_stop_patience:
                        LOGGER.info(
                            f"Early stopping triggered at step {agent.step}. "
                            f"Best FQE LCB = {best_fqe_lcb:.4f}"
                        )
                        break

            LOGGER.info("=" * 60)

        # Regular checkpointing
        if agent.step % cfg.train.save_every == 0 and agent.step > 0:
            agent.save(run_dir / f"checkpoint_{agent.step}.pt")

    # Final save (also happens if early-stopped)
    agent.save(run_dir / "final_model.pt")
    LOGGER.info("Training complete!")

    return run_dir


# =============================================================================
# Part 7: Full Evaluation
# =============================================================================

def evaluate(cfg: ExperimentConfig, checkpoint_path: Path, split: str = "test"):
    """Full evaluation with all OPE methods."""
    # Setup
    set_seed(cfg.train.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Data
    LOGGER.info("Loading data...")
    train_loader, val_loader, test_loader, scaler, spec = make_dataloaders(cfg.data, cfg.train.seed)
    
    if split == "val":
        eval_loader = val_loader
    else:
        eval_loader = test_loader
    
    LOGGER.info(f"Evaluating on {split} set ({len(eval_loader.dataset)} samples)")
    
    # Model
    model = SepsisIQLModel(spec, cfg.encoder, cfg.aux)
    agent = PureIQLAgent(model, cfg.iql, cfg.train, cfg.aux, device)
    agent.load(checkpoint_path)
    LOGGER.info(f"Loaded checkpoint: {checkpoint_path}")
    
    results = {}
    
    # 1. Collapse metrics
    LOGGER.info("Computing collapse metrics...")
    collapse = agent.compute_collapse_metrics(eval_loader)
    results["collapse"] = collapse
    LOGGER.info(f"  entropy={collapse['entropy']:.3f}, top1={collapse['top1_prob']:.3f}")
    
    # 2. FQE
    LOGGER.info("Running FQE...")
    fqe = FQEEstimator(cfg.encoder.d_state, spec.n_actions, cfg.iql.gamma, device)
    fqe.fit(agent, train_loader, cfg.ope)
    fqe_results = fqe.estimate(agent, eval_loader, cfg.ope.bootstrap_n)
    results["fqe"] = fqe_results
    LOGGER.info(f"  FQE: {fqe_results['fqe_mean']:.4f} [{fqe_results['fqe_lcb']:.4f}, {fqe_results['fqe_ucb']:.4f}]")
    
    # 3. WIS (with behavior policy)
    LOGGER.info("Running WIS...")
    behavior = BehaviorPolicyEstimator(cfg.encoder.d_state, spec.n_actions, device)
    behavior.prepare_for_ope(agent.model, train_loader, cfg.ope)
    
    wis_estimator = WISEstimator(cfg.iql.gamma, cfg.ope)
    wis_results = wis_estimator.estimate_with_bootstrap(agent, behavior, eval_loader, device)
    results["wis"] = wis_results
    LOGGER.info(f"  WIS: {wis_results['wis_mean']:.4f} [{wis_results['wis_lcb']:.4f}, {wis_results['wis_ucb']:.4f}]")
    LOGGER.info(f"  ESS: {wis_results['ess']:.1f}, max_ratio: {wis_results['max_ratio']:.1f}")
    
    # 4. Dose gap analysis
    LOGGER.info("Computing dose gap analysis...")
    gap_results = compute_dose_gap_analysis(agent, eval_loader, device)
    results["dose_gap"] = gap_results
    LOGGER.info(f"  Agreement: {gap_results['overall_agreement']*100:.1f}%")
    if "agree_mortality" in gap_results:
        LOGGER.info(f"  Agree mortality: {gap_results['agree_mortality']*100:.1f}%")
    if "disagree_mortality" in gap_results:
        LOGGER.info(f"  Disagree mortality: {gap_results['disagree_mortality']*100:.1f}%")
    
    # Save results
    output_path = checkpoint_path.parent / f"eval_{split}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info(f"Results saved to {output_path}")
    
    return results


# =============================================================================
# Part 8: CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sepsis IQL Unified")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train IQL agent")
    train_parser.add_argument("--data_dir", type=str, default="./processed_data_v2")
    train_parser.add_argument("--run_dir", type=str, default="./runs")
    train_parser.add_argument("--run_name", type=str, default="")
    train_parser.add_argument("--device", type=str, default="cuda")

    # IQL hyperparameters
    train_parser.add_argument("--expectile", type=float, default=0.7)
    train_parser.add_argument("--awbc_beta", type=float, default=3.0)
    train_parser.add_argument("--awbc_clip", type=float, default=20.0)
    train_parser.add_argument("--gamma", type=float, default=0.99)

    # Optional stabilizers
    train_parser.add_argument("--use_entropy_stabilizer", action="store_true")
    train_parser.add_argument("--entropy_stabilizer_coef", type=float, default=0.01)

    # Optional target Q stabilizer
    train_parser.add_argument("--use_target_q", action="store_true")

    # Training parameters
    train_parser.add_argument("--total_steps", type=int, default=200000)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--lr_critic", type=float, default=3e-4)
    train_parser.add_argument("--lr_actor", type=float, default=1e-4)
    train_parser.add_argument("--critic_warmup_steps", type=int, default=10000)
    train_parser.add_argument("--actor_update_every", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=42)

    # Cost control for training-time FQE
    train_parser.add_argument("--fqe_eval_every", type=int, default=10000)
    train_parser.add_argument("--fqe_steps_train", type=int, default=5000)

    # Architecture
    train_parser.add_argument("--d_hidden", type=int, default=128)
    train_parser.add_argument("--d_state", type=int, default=128)
    train_parser.add_argument("--disable_text_fusion", action="store_true")
    train_parser.add_argument("--disable_mnar", action="store_true")
    train_parser.add_argument("--disable_mnar_fusion", action="store_true")
    train_parser.add_argument("--d_mnar_embed", type=int, default=32)

    # Auxiliary losses
    train_parser.add_argument("--disable_aux_outcome", action="store_true")
    train_parser.add_argument("--disable_aux_recon", action="store_true")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained agent")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    eval_parser.add_argument("--data_dir", type=str, default="./processed_data_v2")
    eval_parser.add_argument("--device", type=str, default="cuda")
    eval_parser.add_argument("--bootstrap_n", type=int, default=2000)
    eval_parser.add_argument("--fqe_steps", type=int, default=20000)

    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "train":
        cfg = ExperimentConfig(
            data=DataConfig(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                compute_mnar_features=not args.disable_mnar,
            ),
            encoder=EncoderConfig(
                d_hidden_1h=args.d_hidden,
                d_state=args.d_state,
                enable_text_fusion=not args.disable_text_fusion,
                enable_mnar_fusion=not args.disable_mnar_fusion,
                d_mnar_embed=args.d_mnar_embed,
            ),
            iql=IQLConfig(
                expectile=args.expectile,
                awbc_beta=args.awbc_beta,
                awbc_clip=args.awbc_clip,
                gamma=args.gamma,
                use_entropy_stabilizer=args.use_entropy_stabilizer,
                entropy_stabilizer_coef=args.entropy_stabilizer_coef,
                use_target_q=args.use_target_q,
            ),
            train=TrainConfig(
                total_steps=args.total_steps,
                lr_critic=args.lr_critic,
                lr_actor=args.lr_actor,
                critic_warmup_steps=args.critic_warmup_steps,
                actor_update_every=args.actor_update_every,
                fqe_eval_every=args.fqe_eval_every,
                fqe_steps_during_training=args.fqe_steps_train,
                seed=args.seed,
            ),
            aux=AuxLossConfig(
                enable_outcome=not args.disable_aux_outcome,
                enable_reconstruction=not args.disable_aux_recon,
            ),
            run_name=args.run_name,
            run_dir=args.run_dir,
            device=args.device,
        )

        run_dir = train(cfg)
        LOGGER.info(f"Training complete. Run directory: {run_dir}")

        # Auto-evaluate on test set (full eval uses cfg.ope.fqe_steps, not training fast steps)
        LOGGER.info("Running final evaluation on test set...")
        best_ckpt = run_dir / "best_model.pt"
        if best_ckpt.exists():
            evaluate(cfg, best_ckpt, split="test")

    elif args.command == "eval":
        ckpt_path = Path(args.checkpoint)
        config_path = ckpt_path.parent / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                cfg_dict = json.load(f)
            cfg = ExperimentConfig(
                data=DataConfig(**cfg_dict.get("data", {})),
                encoder=EncoderConfig(**cfg_dict.get("encoder", {})),
                iql=IQLConfig(**cfg_dict.get("iql", {})),
                train=TrainConfig(**cfg_dict.get("train", {})),
                aux=AuxLossConfig(**cfg_dict.get("aux", {})),
                ope=OPEConfig(**cfg_dict.get("ope", {})),
                device=args.device,
            )
        else:
            cfg = ExperimentConfig(
                data=DataConfig(data_dir=args.data_dir),
                device=args.device,
            )

        # Override OPE settings
        cfg.ope.bootstrap_n = args.bootstrap_n
        cfg.ope.fqe_steps = args.fqe_steps
        cfg.data.data_dir = args.data_dir

        evaluate(cfg, ckpt_path, args.split)

    else:
        print("Usage: python sepsis_iql_unified.py {train|eval} [options]")
        print("Run 'python sepsis_iql_unified.py train --help' for training options")
        print("Run 'python sepsis_iql_unified.py eval --help' for evaluation options")


if __name__ == "__main__":
    main()