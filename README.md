# MNAR-RL: Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness

Official implementation for **"Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness"**.

## Overview

MNAR-RL is an offline reinforcement learning framework for clinical decision-making that explicitly leverages **informative missingness** patterns in multimodal electronic health records. Unlike prior approaches that treat missing data as noise, we observe that observation patterns themselves are predictive: sicker patients are monitored more frequently, creating missing-not-at-random (MNAR) signals that correlate with outcomes.

### Key Contributions

1. **MNAR-Aware Multimodal Encoder**: Extends GRU-D with explicit missingness features (time gaps, observation counts, missing rates, windowed frequency) and cross-attention fusion for sparse clinical text.

2. **Action-Conditioned Latent Dynamics**: Theoretical analysis showing that action-independent dynamics lead to vanishing gradients for multi-step credit assignment; our formulation enables learning from delayed terminal rewards.

3. **State Verification via Reconstruction**: Auxiliary objectives ensuring learned representations preserve informative observation patterns.

4. **Comprehensive Evaluation**: Validated on MIMIC-IV (32,837 patients) and eICU (24,562 patients), demonstrating 20.3% improvement over clinician behavior and outperforming CQL by 50.0%.

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (recommended for training)

### Setup

```bash
# Clone repository
git clone https://github.com/anonymous/mnar-rl.git
cd mnar-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Create `requirements.txt`:

```
numpy>=1.21.0
pandas>=1.3.0
torch>=2.0.0
scikit-learn>=1.0.0
transformers>=4.20.0
google-cloud-bigquery>=3.0.0  # Optional: for MIMIC-IV data extraction
pyarrow>=8.0.0
tqdm>=4.62.0
```

## Data Access

This project uses the **MIMIC-IV** database, which requires:

1. **PhysioNet Credentialed Access**: Complete the required training at [PhysioNet](https://physionet.org/content/mimiciv/2.2/)
2. **Google Cloud Platform Setup** (recommended): MIMIC-IV is available on BigQuery for efficient querying

### Option A: BigQuery Access (Recommended)

```bash
# Authenticate with GCP
gcloud auth application-default login

# Set your billing project
export GCP_PROJECT="your-gcp-project-id"
```

### Option B: Local CSV Files

Download MIMIC-IV tables and place in `./sepsis_data/`:
- `sepsis_cohort.csv` (from derived tables)
- `admissions.csv`
- `patients.csv`
- `diagnoses.csv`

## Data Preprocessing

The preprocessing pipeline extracts sepsis cohorts, aligns multimodal observations, and generates training-ready tensors.

### Quick Start

```bash
# With BigQuery (recommended)
python data_preprocessing.py \
    --gcp_billing_project YOUR_GCP_PROJECT \
    --output_dir ./processed_data_v2 \
    --text_backend clinicalbert \
    --text_bert_device cuda

# Without BigQuery (local CSVs only)
python data_preprocessing.py \
    --no_bigquery \
    --sepsis_data_dir ./sepsis_data \
    --output_dir ./processed_data_v2 \
    --text_backend hashing
```

### Full Preprocessing Options

```bash
python data_preprocessing.py \
    --sepsis_data_dir ./sepsis_data \
    --output_dir ./processed_data_v2 \
    --cache_dir ./processed_data_v2/cache \
    --gcp_billing_project YOUR_GCP_PROJECT \
    --mimic_project physionet-data \
    --mimic_version_prefix mimiciv_3_1 \
    --text_backend clinicalbert \
    --text_bert_model "emilyalsentzer/Bio_ClinicalBERT" \
    --text_bert_device cuda \
    --survival_reward 1.0 \
    --death_reward -1.0 \
    --shaping_alpha 0.0 \
    --run_sanity_checks
```

### Preprocessing Output

The pipeline generates the following files in `./processed_data_v2/`:

| File | Shape | Description |
|------|-------|-------------|
| `Y.npy` | (N, 73, 16) | Forward-filled observations (vitals + labs) |
| `mask.npy` | (N, 73, 16) | Observation masks (1=observed, 0=missing) |
| `delta.npy` | (N, 73, 16) | Time since last observation per variable |
| `time_mask.npy` | (N, 73) | Valid time steps per patient |
| `X_static.npy` | (N, 3) | Static features (age, gender, Charlson) |
| `a_4h.npy` | (N, 18) | Discrete actions (9 classes) |
| `r_4h.npy` | (N, 18) | Rewards (terminal only: +1/-1) |
| `done_4h.npy` | (N, 18) | Episode termination flags |
| `step_mask_4h.npy` | (N, 18) | Valid decision steps |
| `e_rad.npy` | (N, 18, 768) | Radiology report embeddings |
| `e_micro.npy` | (N, 18, 768) | Microbiology result embeddings |
| `m_text.npy` | (N, 18, 2) | Text modality availability |
| `metadata.json` | - | Dataset configuration and statistics |

### Text Embedding Backends

| Backend | Dimension | Description |
|---------|-----------|-------------|
| `clinicalbert` | 768 | Bio_ClinicalBERT embeddings (recommended) |
| `hashing` | 256 | HashingVectorizer (no GPU required) |

## Training

### Quick Start

```bash
python sepsis_iql_unified.py train \
    --data_dir ./processed_data_v2 \
    --run_dir ./runs \
    --run_name mnar_rl_exp1 \
    --device cuda
```

### Full Training Configuration

```bash
python sepsis_iql_unified.py train \
    --data_dir ./processed_data_v2 \
    --run_dir ./runs \
    --run_name mnar_rl_full \
    --device cuda \
    \
    # IQL Hyperparameters
    --expectile 0.7 \
    --awbc_beta 3.0 \
    --awbc_clip 20.0 \
    --gamma 0.99 \
    \
    # Training Schedule
    --total_steps 200000 \
    --batch_size 64 \
    --lr_critic 3e-4 \
    --lr_actor 1e-4 \
    --critic_warmup_steps 10000 \
    --actor_update_every 2 \
    \
    # Architecture
    --d_hidden 128 \
    --d_state 128 \
    --d_mnar_embed 32 \
    \
    # Evaluation
    --fqe_eval_every 10000 \
    --fqe_steps_train 5000 \
    \
    --seed 42
```

### Training Phases

The training procedure consists of three phases:

1. **Encoder Pretraining** (steps 0–5,000): Train encoder with auxiliary losses only
2. **Encoder Frozen** (steps 5,000–50,000): Freeze encoder, train RL heads
3. **Joint Finetuning** (steps 50,000+): Unfreeze encoder with reduced learning rate

### Ablation Experiments

```bash
# Disable MNAR features
python sepsis_iql_unified.py train --disable_mnar --run_name ablation_no_mnar

# Disable MNAR fusion in encoder
python sepsis_iql_unified.py train --disable_mnar_fusion --run_name ablation_no_mnar_fusion

# Disable text fusion
python sepsis_iql_unified.py train --disable_text_fusion --run_name ablation_no_text

# Disable auxiliary outcome loss
python sepsis_iql_unified.py train --disable_aux_outcome --run_name ablation_no_outcome
```

### Monitoring Training

Training logs include:
- **Loss components**: Q-loss, V-loss, Actor-loss, Auxiliary losses
- **Policy metrics**: Entropy, top-1 probability, unique actions
- **FQE estimates**: Mean, LCB, UCB with bootstrap confidence intervals

```
[INFO] Step 10000/200000 (45.2 it/s) | phase=freeze_encoder | Q=0.0234 | V=0.0156 | π=0.0089 | H=1.823
[INFO] FQE-fast: mean=0.5234 [0.4891, 0.5577]
```

## Evaluation

### Full Evaluation Pipeline

```bash
python sepsis_iql_unified.py eval \
    --checkpoint ./runs/mnar_rl_exp1/best_model.pt \
    --data_dir ./processed_data_v2 \
    --split test \
    --device cuda \
    --bootstrap_n 2000 \
    --fqe_steps 20000
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **FQE** | Fitted Q-Evaluation with bootstrap CI |
| **WIS** | Weighted Importance Sampling |
| **ESS** | Effective Sample Size |
| **Dose Gap** | Agreement analysis following AI Clinician |

### Output

Evaluation results are saved to `eval_test_results.json`:

```json
{
  "fqe": {
    "fqe_mean": 0.627,
    "fqe_lcb": 0.612,
    "fqe_ucb": 0.641
  },
  "wis": {
    "wis_mean": 0.712,
    "wis_lcb": 0.571,
    "wis_ucb": 0.853,
    "ess": 6.3
  },
  "dose_gap": {
    "overall_agreement": 0.023,
    "agree_mortality": 0.102,
    "disagree_mortality": 0.204
  }
}
```

## Project Structure

```
mnar-rl/
├── data_preprocessing.py    # MIMIC-IV data extraction and preprocessing
├── sepsis_iql_unified.py    # Training and evaluation pipeline
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── sepsis_data/            # Raw MIMIC-IV CSVs (user-provided)
│   ├── sepsis_cohort.csv
│   ├── admissions.csv
│   ├── patients.csv
│   └── diagnoses.csv
│
├── processed_data_v2/      # Preprocessed tensors (generated)
│   ├── *.npy               # Training arrays
│   ├── metadata.json       # Dataset configuration
│   ├── scaler.npz          # Feature normalization parameters
│   └── cache/              # BigQuery cache files
│
└── runs/                   # Training outputs (generated)
    └── <run_name>/
        ├── config.json     # Experiment configuration
        ├── best_model.pt   # Best checkpoint (by FQE LCB)
        ├── final_model.pt  # Final checkpoint
        └── eval_test_results.json
```

## Model Architecture

### Multimodal Encoder

```
Input (1h grid):
  Y: (B, T=73, D=16)     # Observations (vitals + labs)
  mask: (B, T, D)        # Observation masks
  delta: (B, T, D)       # Time since last observation
  mnar_feat: (B, T, 64)  # MNAR features (D×4)
  
GRU-D Cell:
  - Decay-to-mean imputation
  - Trainable decay parameters
  - MNAR feature fusion
  
Decision State Projection (4h grid):
  - Select hidden states at decision times
  - Project to d_state dimensions
  
Text Fusion (optional):
  - Cross-attention over radiology + microbiology
  - Adaptive gating based on availability

Output:
  states: (B, K=18, d_state=128)
```

### IQL Components

```
Q-Networks (×2):  states → Q(s,a) for all actions
Value Network:    states → V(s)
Actor Network:    states → π(a|s) logits
```

## Reproducibility

To reproduce main results:

```bash
# Preprocess data
python data_preprocessing.py \
    --gcp_billing_project YOUR_PROJECT \
    --text_backend clinicalbert \
    --run_sanity_checks

# Train with 5 seeds
for seed in 42 43 44 45 46; do
    python sepsis_iql_unified.py train \
        --seed $seed \
        --run_name mnar_rl_seed${seed}
done

# Evaluate best models
for seed in 42 43 44 45 46; do
    python sepsis_iql_unified.py eval \
        --checkpoint ./runs/mnar_rl_seed${seed}/best_model.pt \
        --split test
done
```

## Troubleshooting

### Common Issues

**BigQuery authentication fails:**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**CUDA out of memory:**
```bash
# Reduce batch size
python sepsis_iql_unified.py train --batch_size 32

# Use CPU for text embeddings
python data_preprocessing.py --text_bert_device cpu
```

**Missing cache files in cache-only mode:**
```bash
# Run once without cache-only to generate cache
python data_preprocessing.py --force_refresh_cache
```
---

**Note**: This is an anonymous submission. Author information and affiliations will be added upon acceptance.
