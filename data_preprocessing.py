import os
import re
import json
import math
import time
import hashlib
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # I/O
    sepsis_data_dir: str = "./sepsis_data"          # local CSVs
    output_dir: str = "./processed_data_v2"         # output folder
    cache_dir: str = "./processed_data_v2/cache"    # query cache

    # BigQuery (optional)
    use_bigquery: bool = True
    gcp_billing_project: str = "your-gcp-project"   # GCP billing/auth project for BigQuery
    mimic_project: str = "physionet-data"           # GCP project containing MIMIC-IV dataset
    mimic_version_prefix: str = "mimiciv_3_1"       # mimiciv_3_1_icu, mimiciv_3_1_hosp, mimiciv_3_1_derived
    mimic_note_dataset: str = "mimiciv_note"        # radiology notes

    # Cache control
    bq_cache_only: bool = False       # if True, never hit BigQuery; require cache files
    force_refresh_cache: bool = False # if True, ignore cache and re-query BigQuery
    print_cache_hits: bool = True     # print whether each query was cache hit
    
    # Time alignment
    horizon_hours: int = 72
    obs_dt_hours: int = 1
    decision_dt_hours: int = 4
    min_observation_hours: int = 6

    # Observation features
    vital_features: Tuple[str, ...] = (
        "heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2", "glucose"
    )
    lab_labels: Tuple[str, ...] = (
        "Glucose", "Lactate", "Potassium", "Sodium", "Creatinine",
        "Platelet Count", "White Blood Cells", "Bilirubin, Total"
    )

    # Treatments: discretization (3 levels each -> 9 actions)
    # If None, cutoffs are learned from cohort distributions (median among >0).
    fluids_high_cutoff_ml: Optional[float] = None
    vaso_high_cutoff_ne: Optional[float] = None

    # Rewards
    survival_reward: float = 1.0
    death_reward: float = -1.0
    shaping_alpha: float = 0.0          # 0 disables SOFA shaping
    shaping_clip: float = 1.0           # clip shaping term to [-clip, clip]

    # Text embeddings
    text_dim: int = 768                 # 768 for ClinicalBERT, 256 for hashing
    text_max_chars: int = 512           # ClinicalBERT max tokens ~512
    text_backend: str = "clinicalbert"  # "clinicalbert" (recommended) or "hashing"
    text_lower: bool = True
    text_ngram_min: int = 1             # only used for hashing backend
    text_ngram_max: int = 2             # only used for hashing backend
    text_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT"  # pretrained model name
    text_bert_batch_size: int = 32      # batch size for BERT encoding
    text_bert_device: str = "cuda"      # "cuda" or "cpu"

    # Performance
    bq_batch_size_stay: int = 3000
    bq_batch_size_hadm: int = 2000
    random_seed: int = 42

    # Sanity checks
    sanity_check_n: int = 200


# =============================================================================
# UTILITIES
# =============================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _to_datetime_safe(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce")

def _clip_text(s: str, max_chars: int) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    if len(s) > max_chars:
        s = s[:max_chars]
    return s

def _clean_text(s: str, lower: bool = True) -> str:
    s = s or ""
    # remove common placeholder patterns and excessive whitespace
    s = re.sub(r"_{2,}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if lower else s

def _safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


# =============================================================================
# OPTIONAL BIGQUERY SUPPORT
# =============================================================================

class BigQueryUnavailable(RuntimeError):
    pass

class BigQueryFetcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = None
        self._bq = None

    def _init_client(self):
        if self._client is not None:
            return
        try:
            from google.cloud import bigquery  # type: ignore
        except Exception as e:
            raise BigQueryUnavailable(
                "google-cloud-bigquery is not installed. Install it (pip install google-cloud-bigquery) "
                "or run with --no_bigquery."
            ) from e
        self._bq = bigquery
        self._client = bigquery.Client(project=self.cfg.gcp_billing_project)

    # Safe helper to create array query parameter
    def array_param(self, name: str, bq_type: str, values: Sequence[Any]) -> Any:
        self._init_client()
        return self._bq.ArrayQueryParameter(name, bq_type, list(values))  # type: ignore

    def query(self, sql: str, cache_key: str, query_params: Optional[List[Any]] = None) -> pd.DataFrame:
        _ensure_dir(self.cfg.cache_dir)
        cache_path = os.path.join(self.cfg.cache_dir, f"{cache_key}.parquet")

        # Expose whether last call hit cache
        self.last_from_cache = False
        self.last_cache_path = cache_path

        # Cache hit
        if (not self.cfg.force_refresh_cache) and os.path.exists(cache_path):
            self.last_from_cache = True
            if self.cfg.print_cache_hits:
                print(f"  [CACHE] {os.path.basename(cache_path)}")
            return pd.read_parquet(cache_path)

        # Cache-only mode: do not hit BigQuery
        if self.cfg.bq_cache_only:
            raise RuntimeError(
                f"Cache miss in --bq_cache_only mode: {cache_path}. "
                f"Either remove --bq_cache_only, or ensure cache exists, or rerun once without cache-only."
            )

        # Query BigQuery
        self._init_client()

        job_config = None
        if query_params is not None:
            job_config = self._bq.QueryJobConfig(query_parameters=query_params)  # type: ignore

        if self.cfg.print_cache_hits:
            print(f"  [BQ] query -> {os.path.basename(cache_path)}")

        df = self._client.query(sql, job_config=job_config).to_dataframe()  # type: ignore
        df.to_parquet(cache_path, index=False)
        return df


# =============================================================================
# TEXT EMBEDDING (DETERMINISTIC, NO TRAINING)
# =============================================================================

class HashingTextEmbedder:
    """
    Deterministic, training-free text embedder based on HashingVectorizer.
    Produces dense float32 vectors of dimension text_dim.
    """
    def __init__(self, text_dim: int, ngram_range: Tuple[int, int] = (1, 2), lowercase: bool = True):
        self.text_dim = int(text_dim)
        self.vectorizer = HashingVectorizer(
            n_features=self.text_dim,
            alternate_sign=False,
            norm=None,
            lowercase=lowercase,
            ngram_range=ngram_range,
            token_pattern=r"(?u)\b\w+\b",
        )

    def encode(self, texts: Sequence[str], batch_size: int = 1024) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.text_dim), dtype=np.float32)

        out = np.zeros((len(texts), self.text_dim), dtype=np.float32)
        start = 0
        while start < len(texts):
            end = min(len(texts), start + batch_size)
            X = self.vectorizer.transform(texts[start:end])  # sparse
            X = X.astype(np.float32)
            dense = X.toarray()
            # L2 normalize (avoid division by zero)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dense = dense / norms
            out[start:end] = dense
            start = end
        return out

class ClinicalBERTEmbedder:
    """
    Semantic text embedder using ClinicalBERT (or other HuggingFace models).
    Produces dense float32 vectors of dimension 768 (BERT hidden size).
    
    Uses [CLS] token representation as the sentence embedding.
    This captures semantic meaning and medical domain knowledge from pretraining.
    """
    def __init__(
        self, 
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "ClinicalBERT backend requires transformers and torch. "
                "Install with: pip install transformers torch"
            )
        
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_length = max_length
        self.batch_size = batch_size
        
        print(f"[ClinicalBERTEmbedder] Loading model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get embedding dimension from model config
        self.text_dim = self.model.config.hidden_size  # 768 for BERT
        print(f"[ClinicalBERTEmbedder] Embedding dimension: {self.text_dim}")
    
    def encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode texts to dense vectors using ClinicalBERT.
        
        Args:
            texts: List of text strings
            batch_size: Override default batch size
            
        Returns:
            np.ndarray of shape (len(texts), 768)
        """
        if len(texts) == 0:
            return np.zeros((0, self.text_dim), dtype=np.float32)
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                if len(texts) >= batch_size * 2:
                    print(f"  BERT encoding: {start}/{len(texts)}", end="\r")
                end = min(len(texts), start + batch_size)
                batch_texts = [str(t) if t else "" for t in texts[start:end]]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                
                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
                
                # L2 normalize for consistency with hashing embedder
                norms = torch.norm(cls_embeddings, dim=1, keepdim=True).clamp(min=1e-8)
                cls_embeddings = cls_embeddings / norms
                
                all_embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)


def create_text_embedder(cfg: Config):
    """
    Factory function to create the appropriate text embedder based on config.
    """
    print(f"[create_text_embedder] cfg.text_backend = {cfg.text_backend}")
    print(f"[create_text_embedder] cfg.text_bert_device = {getattr(cfg, 'text_bert_device', 'NOT SET')}")
    
    if cfg.text_backend == "clinicalbert":
        if not TRANSFORMERS_AVAILABLE:
            print("[WARNING] transformers not available, falling back to hashing backend")
            return HashingTextEmbedder(
                text_dim=256,
                ngram_range=(cfg.text_ngram_min, cfg.text_ngram_max),
                lowercase=cfg.text_lower,
            )
        return ClinicalBERTEmbedder(
            model_name=cfg.text_bert_model,
            device=cfg.text_bert_device,
            max_length=cfg.text_max_chars,
            batch_size=cfg.text_bert_batch_size,
        )
    else:
        # Default: hashing
        return HashingTextEmbedder(
            text_dim=cfg.text_dim,
            ngram_range=(cfg.text_ngram_min, cfg.text_ngram_max),
            lowercase=cfg.text_lower,
        )

# =============================================================================
# LOAD LOCAL FILES
# =============================================================================

def load_local_csv(cfg: Config, filename: str) -> pd.DataFrame:
    path = os.path.join(cfg.sepsis_data_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def load_local_inputs(cfg: Config) -> Dict[str, pd.DataFrame]:
    data = {
        "sepsis_cohort": load_local_csv(cfg, "sepsis_cohort.csv"),
        "admissions": load_local_csv(cfg, "admissions.csv"),
        "patients": load_local_csv(cfg, "patients.csv"),
        "diagnoses": load_local_csv(cfg, "diagnoses.csv"),
        # optional / fallback
        "vitals_local": load_local_csv(cfg, "vitals.csv"),
        "labs_first_48h": load_local_csv(cfg, "labs_first_48h.csv"),
        "radiology_notes_local": load_local_csv(cfg, "radiology_notes.csv"),
    }
    return data


# =============================================================================
# COHORT BUILDING
# =============================================================================

def build_master_cohort(local: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    cohort = local["sepsis_cohort"].copy()
    if cohort.empty:
        raise RuntimeError("Missing sepsis_cohort.csv in sepsis_data_dir.")

    # parse times
    cohort["suspected_infection_time"] = cohort["suspected_infection_time"].apply(_to_datetime_safe)
    cohort["t0"] = cohort["suspected_infection_time"]

    # filter sepsis3 if present
    if "sepsis3" in cohort.columns:
        cohort = cohort[cohort["sepsis3"].astype(bool)].copy()

    # admissions merge
    adm = local["admissions"].copy()
    for c in ["admittime", "dischtime", "deathtime"]:
        if c in adm.columns:
            adm[c] = adm[c].apply(_to_datetime_safe)
    cohort = cohort.merge(
        adm[["hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag",
             "readmission_30d", "los_hospital_days"]],
        on="hadm_id", how="left"
    )

    # patients merge
    pat = local["patients"].copy()
    cohort = cohort.merge(pat[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")

    # observation end time for this 72h window
    cohort["max_obs_time"] = cohort["t0"] + pd.to_timedelta(cfg.horizon_hours, unit="h")
    cohort["event_time"] = cohort["deathtime"].fillna(cohort["dischtime"])
    cohort["observation_end"] = cohort[["event_time", "max_obs_time"]].min(axis=1)
    cohort["observation_hours"] = (cohort["observation_end"] - cohort["t0"]).dt.total_seconds() / 3600.0

    # filter invalid
    cohort = cohort[cohort["t0"].notna()].copy()
    cohort = cohort[(cohort["observation_hours"] >= cfg.min_observation_hours)].copy()

    # outcomes (robust)
    # Some rows may have hospital_expire_flag==1 but missing deathtime (or vice versa).
    # Treat as in-hospital death if either indicates death.
    flag = cohort.get("hospital_expire_flag", pd.Series([0] * len(cohort))).fillna(0).astype(int)
    has_death_time = cohort.get("deathtime", pd.Series([pd.NaT] * len(cohort))).notna().astype(int)
    cohort["mortality_in_hosp"] = ((flag == 1) | (has_death_time == 1)).astype(int)

    cohort["death_hours"] = np.where(
        cohort["deathtime"].notna(),
        (cohort["deathtime"] - cohort["t0"]).dt.total_seconds() / 3600.0,
        np.nan
    )

    # clip to horizon for episode end
    cohort["episode_end_hours"] = np.minimum(cohort["observation_hours"].values, float(cfg.horizon_hours))

    cohort = cohort.sort_values("stay_id").reset_index(drop=True)
    return cohort


# =============================================================================
# CHARLSON COMORBIDITY (simple ICD-10 prefix matching)
# =============================================================================

def compute_charlson_features(diagnoses: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    if diagnoses.empty:
        out = cohort[["hadm_id", "stay_id"]].copy()
        out["charlson_index"] = 0.0
        return out

    dx = diagnoses.copy()
    dx["icd_code"] = dx["icd_code"].astype(str)

    charlson_categories = {
        "mi": ["I21", "I22", "I25.2"],
        "chf": ["I50", "I11.0", "I13.0", "I13.2"],
        "pvd": ["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1", "I79.0"],
        "stroke": ["G45", "G46", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
        "dementia": ["F00", "F01", "F02", "F03", "G30", "G31.1"],
        "copd": ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"],
        "rheumatic": ["M05", "M06", "M32", "M33", "M34"],
        "peptic_ulcer": ["K25", "K26", "K27", "K28"],
        "liver_mild": ["K70.0", "K70.1", "K70.2", "K70.3", "K70.9", "K71", "K73", "K74", "K76.0"],
        "diabetes_uncomplicated": ["E10.0", "E10.1", "E10.9", "E11.0", "E11.1", "E11.9", "E13.0", "E13.1", "E13.9"],
        "diabetes_complicated": ["E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8",
                                 "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8"],
        "paraplegia": ["G04.1", "G11.4", "G80.1", "G80.2", "G81", "G82"],
        "renal": ["N18.3", "N18.4", "N18.5", "N18.6", "N19", "N25.0", "Z49", "Z94.0", "Z99.2"],
        "cancer": ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
                   "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
                   "C20", "C21", "C22", "C23", "C24", "C25", "C26"],
        "liver_severe": ["K70.4", "K72", "K76.6", "I85.0", "I85.9", "I86.4", "I98.2"],
        "metastatic": ["C77", "C78", "C79", "C80"],
        "aids": ["B20", "B21", "B22", "B23", "B24"],
    }
    weights = {
        "mi": 1, "chf": 1, "pvd": 1, "stroke": 1, "dementia": 1, "copd": 1,
        "rheumatic": 1, "peptic_ulcer": 1, "liver_mild": 1,
        "diabetes_uncomplicated": 1, "diabetes_complicated": 2,
        "paraplegia": 2, "renal": 2, "cancer": 2,
        "liver_severe": 3, "metastatic": 6, "aids": 6
    }

    hadm_ids = cohort["hadm_id"].unique().tolist()
    dx = dx[dx["hadm_id"].isin(hadm_ids)].copy()

    # pre-group codes per hadm
    codes_by_hadm = dx.groupby("hadm_id")["icd_code"].apply(list).to_dict()

    out = cohort[["hadm_id", "stay_id"]].drop_duplicates().copy()
    for cat, prefixes in charlson_categories.items():
        def f(hadm_id: int) -> int:
            codes = codes_by_hadm.get(hadm_id, [])
            for c in codes:
                for p in prefixes:
                    if c.startswith(p):
                        return 1
            return 0
        out[f"charlson_{cat}"] = out["hadm_id"].apply(f).astype(int)

    out["charlson_index"] = 0.0
    for cat, w in weights.items():
        out["charlson_index"] += out[f"charlson_{cat}"].astype(float) * float(w)

    return out


# =============================================================================
# BIGQUERY EXTRACTION
# =============================================================================

def _fq(cfg: Config, dataset: str, table: str) -> str:
    return f"`{cfg.mimic_project}.{dataset}.{table}`"

def fetch_vitals_bq(fetcher: BigQueryFetcher, cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ds = f"{cfg.mimic_version_prefix}_derived"
    table = "vitalsign"
    stay_ids = cohort["stay_id"].astype(int).tolist()

    out: List[pd.DataFrame] = []
    bs = cfg.bq_batch_size_stay
    for i in range(0, len(stay_ids), bs):
        batch = [int(x) for x in stay_ids[i:i+bs]]

        sql = f"""
        SELECT stay_id, charttime,
            heart_rate, sbp, dbp, mbp, resp_rate, temperature, spo2, glucose
        FROM {_fq(cfg, ds, table)}
        WHERE stay_id IN UNNEST(@stay_ids)
        ORDER BY stay_id, charttime
        """

        cache_key = f"vitals_{i//bs}_{_sha1(','.join(map(str, batch)))[:10]}"
        params = [fetcher.array_param("stay_ids", "INT64", batch)]
        out.append(fetcher.query(sql, cache_key, query_params=params))
        print(f"  [BQ] vitals batch {i//bs+1}: {len(out[-1]):,} rows")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    df["charttime"] = df["charttime"].apply(_to_datetime_safe)
    return df

def fetch_labs_bq(fetcher: BigQueryFetcher, cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    IMPORTANT: For RL state, labs must be aligned by availability time (storetime) to avoid future leakage.
    We fetch both charttime and storetime.
    """
    ds_hosp = f"{cfg.mimic_version_prefix}_hosp"
    dlab = "d_labitems"
    labels = list(cfg.lab_labels)
    labels_sql = ",".join("'" + str(l).replace("'", "''") + "'" for l in labels)

    sql_items = f"""
    SELECT itemid, label
    FROM {_fq(cfg, ds_hosp, dlab)}
    WHERE label IN ({labels_sql})
    """
    items = fetcher.query(sql_items, f"labitems_{_sha1(labels_sql)[:10]}")
    if items.empty:
        return pd.DataFrame()

    itemids = items["itemid"].astype(int).tolist()
    itemids_str = ",".join(map(str, itemids))

    hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()
    out: List[pd.DataFrame] = []
    bs = cfg.bq_batch_size_hadm

    for i in range(0, len(hadm_ids), bs):
        batch = hadm_ids[i:i+bs]
        ids_str = ",".join(map(str, batch))
        sql = f"""
        SELECT
          le.hadm_id,
          le.itemid,
          di.label AS lab_name,
          le.charttime,
          le.storetime,
          le.valuenum
        FROM {_fq(cfg, ds_hosp, "labevents")} le
        JOIN {_fq(cfg, ds_hosp, "d_labitems")} di
          ON le.itemid = di.itemid
        WHERE le.hadm_id IN ({ids_str})
          AND le.itemid IN ({itemids_str})
          AND le.valuenum IS NOT NULL
        ORDER BY le.hadm_id, le.charttime
        """
        cache_key = f"labs_{i//bs}_{_sha1(ids_str)[:10]}"
        df_batch = fetcher.query(sql, cache_key)  # 或带 query_params 的版本
        out.append(df_batch)

        tag = "CACHE" if getattr(fetcher, "last_from_cache", False) else "BQ"
        print(f"  [{tag}] vitals batch {i//bs+1}: {len(df_batch):,} rows")

    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if df.empty:
        return df

    for c in ["charttime", "storetime"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_datetime_safe)

    return df

def fetch_sofa_bq(fetcher: BigQueryFetcher, cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ds = f"{cfg.mimic_version_prefix}_derived"
    table = "sofa"
    stay_ids = cohort["stay_id"].astype(int).tolist()

    out: List[pd.DataFrame] = []
    bs = cfg.bq_batch_size_stay
    for i in range(0, len(stay_ids), bs):
        batch = stay_ids[i:i+bs]
        ids_str = ",".join(map(str, batch))
        sql = f"""
        SELECT stay_id, starttime, endtime,
               respiration, coagulation, liver, cardiovascular, cns, renal, sofa_24hours
        FROM {_fq(cfg, ds, table)}
        WHERE stay_id IN ({ids_str})
        ORDER BY stay_id, starttime
        """
        cache_key = f"sofa_{i//bs}_{_sha1(ids_str)[:10]}"
        out.append(fetcher.query(sql, cache_key))
        print(f"  [BQ] sofa batch {i//bs+1}: {len(out[-1]):,} rows")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    df["starttime"] = df["starttime"].apply(_to_datetime_safe)
    df["endtime"] = df["endtime"].apply(_to_datetime_safe)
    return df

def fetch_ne_equiv_bq(fetcher: BigQueryFetcher, cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ds = f"{cfg.mimic_version_prefix}_derived"
    table = "norepinephrine_equivalent_dose"
    stay_ids = cohort["stay_id"].astype(int).tolist()

    out: List[pd.DataFrame] = []
    bs = cfg.bq_batch_size_stay
    for i in range(0, len(stay_ids), bs):
        batch = stay_ids[i:i+bs]
        ids_str = ",".join(map(str, batch))
        sql = f"""
        SELECT stay_id, starttime, endtime, norepinephrine_equivalent_dose
        FROM {_fq(cfg, ds, table)}
        WHERE stay_id IN ({ids_str})
        ORDER BY stay_id, starttime
        """
        cache_key = f"ne_{i//bs}_{_sha1(ids_str)[:10]}"
        out.append(fetcher.query(sql, cache_key))
        print(f"  [BQ] ne_equiv batch {i//bs+1}: {len(out[-1]):,} rows")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    df["starttime"] = df["starttime"].apply(_to_datetime_safe)
    df["endtime"] = df["endtime"].apply(_to_datetime_safe)
    # numeric
    if "norepinephrine_equivalent_dose" in df.columns:
        df["norepinephrine_equivalent_dose"] = pd.to_numeric(df["norepinephrine_equivalent_dose"], errors="coerce")
    return df

def fetch_fluids_bq(fetcher: BigQueryFetcher, cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ds = f"{cfg.mimic_version_prefix}_icu"
    table = "inputevents"
    stay_ids = cohort["stay_id"].astype(int).tolist()

    out: List[pd.DataFrame] = []
    bs = cfg.bq_batch_size_stay
    for i in range(0, len(stay_ids), bs):
        batch = stay_ids[i:i+bs]
        ids_str = ",".join(map(str, batch))
        sql = f"""
        SELECT stay_id, starttime, endtime, amount, amountuom,
               itemid, ordercategoryname, ordercategorydescription
        FROM {_fq(cfg, ds, table)}
        WHERE stay_id IN ({ids_str})
          AND amount IS NOT NULL
          AND (LOWER(ordercategoryname) = 'fluids' OR LOWER(ordercategorydescription) LIKE '%fluid%' OR LOWER(ordercategorydescription) LIKE '%bolus%')
        ORDER BY stay_id, starttime
        """
        cache_key = f"fluids_{i//bs}_{_sha1(ids_str)[:10]}"
        out.append(fetcher.query(sql, cache_key))
        print(f"  [BQ] fluids batch {i//bs+1}: {len(out[-1]):,} rows")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    df["starttime"] = df["starttime"].apply(_to_datetime_safe)
    df["endtime"] = df["endtime"].apply(_to_datetime_safe)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["amountuom"] = df["amountuom"].astype(str)
    return df

def fetch_radiology_bq(fetcher: BigQueryFetcher, cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ds = cfg.mimic_note_dataset
    table = "radiology"
    hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()

    out: List[pd.DataFrame] = []
    bs = cfg.bq_batch_size_hadm
    for i in range(0, len(hadm_ids), bs):
        batch = hadm_ids[i:i+bs]
        ids_str = ",".join(map(str, batch))
        sql = f"""
        SELECT note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
        FROM {_fq(cfg, ds, table)}
        WHERE hadm_id IN ({ids_str})
        ORDER BY hadm_id, charttime
        """
        cache_key = f"rad_{i//bs}_{_sha1(ids_str)[:10]}"
        out.append(fetcher.query(sql, cache_key))
        print(f"  [BQ] radiology batch {i//bs+1}: {len(out[-1]):,} rows")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    for c in ["charttime", "storetime"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_datetime_safe)
    return df

def fetch_micro_bq(fetcher: BigQueryFetcher, cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ds = f"{cfg.mimic_version_prefix}_hosp"
    table = "microbiologyevents"
    hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()

    out: List[pd.DataFrame] = []
    bs = cfg.bq_batch_size_hadm
    for i in range(0, len(hadm_ids), bs):
        batch = hadm_ids[i:i+bs]
        ids_str = ",".join(map(str, batch))
        sql = f"""
        SELECT microevent_id, subject_id, hadm_id,
               charttime, storetime, spec_type_desc, test_name, org_name,
               interpretation, comments, ab_name
        FROM {_fq(cfg, ds, table)}
        WHERE hadm_id IN ({ids_str})
        ORDER BY hadm_id, charttime
        """
        cache_key = f"micro_{i//bs}_{_sha1(ids_str)[:10]}"
        out.append(fetcher.query(sql, cache_key))
        print(f"  [BQ] micro batch {i//bs+1}: {len(out[-1]):,} rows")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    for c in ["charttime", "storetime"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_datetime_safe)
    return df


# =============================================================================
# TIME ALIGNMENT HELPERS
# =============================================================================

def add_hours_from_t0(
    df: pd.DataFrame,
    cohort: pd.DataFrame,
    time_col: str,
    join_col: str,
    key_col_in_cohort: str,
) -> pd.DataFrame:
    """
    Adds columns:
      - t0 (mapped from cohort)
      - hours_from_t0 (float)
      - hour_bin (int floor)
    """
    if df.empty:
        return df
    t0_map = cohort.set_index(key_col_in_cohort)["t0"].to_dict()
    df = df.copy()
    df["t0"] = df[join_col].map(t0_map)
    df = df[df["t0"].notna()].copy()
    df[time_col] = df[time_col].apply(_to_datetime_safe)
    df["hours_from_t0"] = (df[time_col] - df["t0"]).dt.total_seconds() / 3600.0
    df = df[df["hours_from_t0"].notna()].copy()
    df["hour_bin"] = np.floor(df["hours_from_t0"]).astype(int)
    return df

def _fixed_time_grids(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    t_1h = np.arange(0, cfg.horizon_hours + cfg.obs_dt_hours, cfg.obs_dt_hours, dtype=np.int32)   # 0..72 inclusive => 73
    t_4h = np.arange(0, cfg.horizon_hours, cfg.decision_dt_hours, dtype=np.int32)                 # 0..68 => 18
    return t_1h, t_4h


# =============================================================================
# BUILD OBSERVATION TENSORS
# =============================================================================

def build_obs_tensors(
    cohort: pd.DataFrame,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    cfg: Config
) -> Dict[str, Any]:
    """
    Builds:
      Y_raw (N, 73, D_obs) with NaNs where missing
      mask  (N, 73, D_obs)
      time_mask (N, 73) 1 within episode window, else 0

    Key fixes:
      - Labs aligned by availability time: storetime if present else charttime.
      - hadm-level tables (labs) mapped to stay via merge (supports multiple stays per hadm).
      - Filter by hours_from_t0 <= horizon (NOT hour_bin <= horizon).
      - Force mask to be zero where time_mask==0.
    """
    t_1h, _ = _fixed_time_grids(cfg)
    T = len(t_1h)
    feat_names = list(cfg.vital_features) + list(cfg.lab_labels)
    D = len(feat_names)
    N = len(cohort)

    stay_ids = cohort["stay_id"].astype(int).tolist()
    stay_to_i = {sid: i for i, sid in enumerate(stay_ids)}

    Y = np.full((N, T, D), np.nan, dtype=np.float32)

    # ---------------- vitals (stay-level) ----------------
    if vitals is not None and (not vitals.empty):
        v = vitals.copy()
        v = add_hours_from_t0(v, cohort, time_col="charttime", join_col="stay_id", key_col_in_cohort="stay_id")
        # strict horizon filter
        v = v[(v["hours_from_t0"] >= 0) & (v["hours_from_t0"] <= cfg.horizon_hours)].copy()
        # aggregate
        v_agg = v.groupby(["stay_id", "hour_bin"], as_index=False)[list(cfg.vital_features)].mean()
        i_idx = v_agg["stay_id"].map(stay_to_i).to_numpy()
        t_idx = v_agg["hour_bin"].to_numpy()
        vals = v_agg[list(cfg.vital_features)].to_numpy(dtype=np.float32)
        Y[i_idx, t_idx, 0:len(cfg.vital_features)] = vals

    # ---------------- labs (hadm-level) ----------------
    if labs is not None and (not labs.empty):
        l = labs.copy()

        # availability time: prefer storetime when present (prevents future leakage)
        if "storetime" in l.columns:
            l["avail_time"] = l["storetime"].fillna(l.get("charttime", pd.NaT))
        else:
            l["avail_time"] = l.get("charttime", pd.NaT)

        l["avail_time"] = l["avail_time"].apply(_to_datetime_safe)

        map_cols = ["hadm_id", "stay_id", "t0"]
        if "episode_end_hours" in cohort.columns:
            map_cols.append("episode_end_hours")

        l = l.merge(cohort[map_cols], on="hadm_id", how="inner")

        l["hours_from_t0"] = (l["avail_time"] - l["t0"]).dt.total_seconds() / 3600.0
        l = l[l["hours_from_t0"].notna()].copy()

        if "episode_end_hours" in l.columns:
            l["max_h"] = np.minimum(float(cfg.horizon_hours), l["episode_end_hours"].astype(float))
            l = l[(l["hours_from_t0"] >= 0) & (l["hours_from_t0"] <= l["max_h"])].copy()
        else:
            l = l[(l["hours_from_t0"] >= 0) & (l["hours_from_t0"] <= float(cfg.horizon_hours))].copy()

        l["hour_bin"] = np.floor(l["hours_from_t0"]).astype(int)

        if "lab_name" not in l.columns:
            if "label" in l.columns:
                l["lab_name"] = l["label"]

        l = l[l["lab_name"].isin(cfg.lab_labels)].copy()
        l["valuenum"] = pd.to_numeric(l["valuenum"], errors="coerce")
        l = l[l["valuenum"].notna()].copy()

        # aggregate
        l_agg = l.groupby(["stay_id", "hour_bin", "lab_name"], as_index=False)["valuenum"].mean()

        for j, lab in enumerate(cfg.lab_labels):
            sub = l_agg[l_agg["lab_name"] == lab]
            if sub.empty:
                continue
            i_idx = sub["stay_id"].map(stay_to_i).to_numpy()
            t_idx = sub["hour_bin"].to_numpy()
            vals = sub["valuenum"].to_numpy(dtype=np.float32)
            Y[i_idx, t_idx, len(cfg.vital_features) + j] = vals

    mask = (~np.isnan(Y)).astype(np.float32)

    # time_mask: 1 for valid hours within episode_end_hours, else 0
    end_hours = cohort["episode_end_hours"].to_numpy(dtype=np.float32)
    time_mask = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        last_valid = int(min(cfg.horizon_hours, math.floor(float(end_hours[i]))))
        time_mask[i, :last_valid + 1] = 1.0

    # enforce no observations beyond episode window
    mask = mask * time_mask[:, :, None]

    return {"Y_raw": Y, "mask": mask, "time_mask": time_mask, "feature_names": feat_names, "t_1h": t_1h}

def forward_fill_and_deltas(Y_raw: np.ndarray, mask: np.ndarray, time_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Forward-fill missing values per patient-feature, but keep mask and compute delta times.

    Returns:
      Y_ffill (float32)
      delta (float32): hours since last observed (0 where observed)
    """
    N, T, D = Y_raw.shape
    Y = Y_raw.copy()
    delta = np.zeros((N, T, D), dtype=np.float32)

    # compute global per-feature mean (for initial imputation)
    means = np.nanmean(Y_raw.reshape(-1, D), axis=0).astype(np.float32)
    means[np.isnan(means)] = 0.0

    for i in range(N):
        for d in range(D):
            last_val = means[d]
            last_obs_t = None
            for t in range(T):
                if time_mask[i, t] == 0:
                    # beyond episode window: keep zeros, masks are 0
                    Y[i, t, d] = 0.0
                    delta[i, t, d] = 0.0
                    continue

                if mask[i, t, d] > 0:
                    last_val = float(Y[i, t, d])
                    last_obs_t = t
                    delta[i, t, d] = 0.0
                else:
                    Y[i, t, d] = last_val
                    if last_obs_t is None:
                        delta[i, t, d] = float(t)  # since start
                    else:
                        delta[i, t, d] = float(t - last_obs_t)

    return {"Y": Y.astype(np.float32), "delta": delta.astype(np.float32), "means": means}


# =============================================================================
# BUILD 4H ACTIONS FROM CONTINUOUS EXPOSURES
# =============================================================================

def _overlap_hours(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> float:
    """Overlap duration in hours between [a_start, a_end] and [b_start, b_end]."""
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    if pd.isna(latest_start) or pd.isna(earliest_end) or earliest_end <= latest_start:
        return 0.0
    return (earliest_end - latest_start).total_seconds() / 3600.0

def build_continuous_exposures(
    cohort: pd.DataFrame,
    fluids: pd.DataFrame,
    ne: pd.DataFrame,
    cfg: Config
) -> Dict[str, np.ndarray]:
    """
    Computes:
      fluids_ml_4h (N, 18): overlap-weighted sum of mL
      vaso_ne_4h   (N, 18): max NE-eq dose during interval

    NOTE: clip all exposures to effective_end = min(horizon, episode_end_hours) to prevent post-terminal actions.
    """
    _, t_4h = _fixed_time_grids(cfg)
    N = len(cohort)
    K = len(t_4h)

    stay_ids = cohort["stay_id"].astype(int).tolist()
    stay_to_i = {sid: i for i, sid in enumerate(stay_ids)}
    t0_map = cohort.set_index("stay_id")["t0"].to_dict()

    if "episode_end_hours" in cohort.columns:
        end_map = cohort.set_index("stay_id")["episode_end_hours"].to_dict()
    else:
        end_map = {sid: float(cfg.horizon_hours) for sid in stay_ids}

    fluids_4h = np.zeros((N, K), dtype=np.float32)
    vaso_4h = np.zeros((N, K), dtype=np.float32)

    def _bin_index_from_time(t0: pd.Timestamp, ts: pd.Timestamp) -> int:
        rel_h = (ts - t0).total_seconds() / 3600.0
        k = int(math.floor(rel_h / float(cfg.decision_dt_hours)))
        return max(0, min(K - 1, k))

    # ---------- fluids ----------
    if fluids is not None and (not fluids.empty):
        f = fluids.copy()
        f = f[f["stay_id"].isin(stay_ids)].copy()
        f["starttime"] = f["starttime"].apply(_to_datetime_safe)
        f["endtime"] = f["endtime"].apply(_to_datetime_safe)

        # if endtime missing, treat as instant (bolus)
        f["endtime"] = f["endtime"].fillna(f["starttime"])
        f["amount"] = pd.to_numeric(f["amount"], errors="coerce")
        f = f[f["amount"].notna()].copy()

        f["amountuom"] = f.get("amountuom", "").astype(str)

        def to_ml(amount: float, uom: str) -> Optional[float]:
            u = (uom or "").strip().lower()
            # ignore suspicious rate-like uoms
            if "/" in u:
                return None
            if u == "" or "ml" in u:
                return float(amount)
            if u in ["l", "liter", "liters"] or (" litre" in u) or (" liter" in u):
                return float(amount) * 1000.0
            # unknown unit -> assume mL (conservative but avoids dropping)
            return float(amount)

        f["amount_ml"] = [
            to_ml(a, u) for a, u in zip(f["amount"].to_numpy(), f["amountuom"].to_numpy())
        ]
        f = f[f["amount_ml"].notna()].copy()
        f["amount_ml"] = f["amount_ml"].astype(float)

        for row in f.itertuples(index=False):
            sid = int(getattr(row, "stay_id"))
            i = stay_to_i.get(sid)
            if i is None:
                continue
            t0 = t0_map.get(sid)
            if pd.isna(t0):
                continue
            s = getattr(row, "starttime")
            e = getattr(row, "endtime")
            if pd.isna(s) or pd.isna(e):
                continue

            amt = float(getattr(row, "amount_ml"))
            if (not np.isfinite(amt)) or (amt <= 0.0):
                continue

            # per-stay effective end time = min(horizon, episode_end_hours)
            end_h = float(end_map.get(sid, float(cfg.horizon_hours)))
            if np.isnan(end_h):
                end_h = float(cfg.horizon_hours)
            end_h = max(0.0, min(float(cfg.horizon_hours), end_h))
            effective_end = t0 + pd.to_timedelta(end_h, unit="h")

            clip_start = max(s, t0)
            clip_end = min(e, effective_end)

            # instant/bolus event: assign all to the bin containing start
            if clip_end <= clip_start:
                if clip_start < t0 or clip_start > effective_end:
                    continue
                k = _bin_index_from_time(t0, clip_start)
                fluids_4h[i, k] += amt
                continue

            total_dur = (clip_end - clip_start).total_seconds() / 3600.0
            total_dur = max(total_dur, 1e-6)

            for k, t0h in enumerate(t_4h):
                bin_start = t0 + pd.to_timedelta(int(t0h), unit="h")
                bin_end = bin_start + pd.to_timedelta(cfg.decision_dt_hours, unit="h")
                ov = _overlap_hours(clip_start, clip_end, bin_start, bin_end)
                if ov > 0:
                    fluids_4h[i, k] += amt * float(ov / total_dur)

    # ---------- vasopressor (NE-eq) ----------
    if ne is not None and (not ne.empty):
        v = ne.copy()
        v = v[v["stay_id"].isin(stay_ids)].copy()
        v["starttime"] = v["starttime"].apply(_to_datetime_safe)
        v["endtime"] = v["endtime"].apply(_to_datetime_safe)
        v["endtime"] = v["endtime"].fillna(v["starttime"])
        v["norepinephrine_equivalent_dose"] = pd.to_numeric(v["norepinephrine_equivalent_dose"], errors="coerce")
        v = v[v["norepinephrine_equivalent_dose"].notna()].copy()

        for row in v.itertuples(index=False):
            sid = int(getattr(row, "stay_id"))
            i = stay_to_i.get(sid)
            if i is None:
                continue
            t0 = t0_map.get(sid)
            if pd.isna(t0):
                continue
            s = getattr(row, "starttime")
            e = getattr(row, "endtime")
            if pd.isna(s) or pd.isna(e):
                continue

            end_h = float(end_map.get(sid, float(cfg.horizon_hours)))
            if np.isnan(end_h):
                end_h = float(cfg.horizon_hours)
            end_h = max(0.0, min(float(cfg.horizon_hours), end_h))
            effective_end = t0 + pd.to_timedelta(end_h, unit="h")

            clip_start = max(s, t0)
            clip_end = min(e, effective_end)
            dose = float(getattr(row, "norepinephrine_equivalent_dose"))

            # instant event
            if clip_end <= clip_start:
                if clip_start < t0 or clip_start > effective_end:
                    continue
                k = _bin_index_from_time(t0, clip_start)
                vaso_4h[i, k] = max(vaso_4h[i, k], dose)
                continue

            for k, t0h in enumerate(t_4h):
                bin_start = t0 + pd.to_timedelta(int(t0h), unit="h")
                bin_end = bin_start + pd.to_timedelta(cfg.decision_dt_hours, unit="h")
                ov = _overlap_hours(clip_start, clip_end, bin_start, bin_end)
                if ov > 0:
                    vaso_4h[i, k] = max(vaso_4h[i, k], dose)

    return {"fluids_ml_4h": fluids_4h, "vaso_ne_4h": vaso_4h, "t_4h": t_4h}


def discretize_actions(
    fluids_ml_4h: np.ndarray,
    vaso_ne_4h: np.ndarray,
    cfg: Config
) -> Dict[str, Any]:
    """
    Discretize continuous fluids and vaso into 3 levels each.
      level 0: ==0
      level 1: (0, cutoff]
      level 2: > cutoff
    cutoff defaults to median among >0 if not specified.
    """
    # compute cutoffs if needed
    fluids_pos = fluids_ml_4h[fluids_ml_4h > 0]
    vaso_pos = vaso_ne_4h[vaso_ne_4h > 0]

    f_cut = cfg.fluids_high_cutoff_ml
    v_cut = cfg.vaso_high_cutoff_ne

    if f_cut is None:
        f_cut = float(np.median(fluids_pos)) if fluids_pos.size > 0 else 0.0
    if v_cut is None:
        v_cut = float(np.median(vaso_pos)) if vaso_pos.size > 0 else 0.0

    # levels
    f_level = np.zeros_like(fluids_ml_4h, dtype=np.int64)
    v_level = np.zeros_like(vaso_ne_4h, dtype=np.int64)

    f_level[(fluids_ml_4h > 0) & (fluids_ml_4h <= f_cut)] = 1
    f_level[(fluids_ml_4h > f_cut)] = 2

    v_level[(vaso_ne_4h > 0) & (vaso_ne_4h <= v_cut)] = 1
    v_level[(vaso_ne_4h > v_cut)] = 2

    a_4h = (f_level * 3 + v_level).astype(np.int64)  # 0..8

    # expand to 1h (72 intervals): hour h uses action of bin h//4
    # for 73 points, there are 72 intervals. We'll store 72 actions.
    N, K = a_4h.shape
    a_1h = np.zeros((N, 72), dtype=np.int64)
    for h in range(72):
        a_1h[:, h] = a_4h[:, h // 4]

    return {
        "a_4h": a_4h,
        "a_1h": a_1h,
        "fluids_level_4h": f_level.astype(np.int64),
        "vaso_level_4h": v_level.astype(np.int64),
        "fluids_cutoff_ml": float(f_cut),
        "vaso_cutoff_ne": float(v_cut),
    }


# =============================================================================
# REWARD / DONE
# =============================================================================

def compute_terminal_step(event_hour: float, cfg: Config) -> int:
    """Map an event at time tau (hours) to the step index whose interval contains tau."""
    if np.isnan(event_hour):
        return 17
    tau = float(event_hour)
    tau = max(0.0, min(float(cfg.horizon_hours), tau))
    # if tau is exactly at boundary, attribute to previous step
    k = int(math.ceil(tau / float(cfg.decision_dt_hours)) - 1)
    k = max(0, min(17, k))
    return k

def build_reward_done(
    cohort: pd.DataFrame,
    sofa: pd.DataFrame,
    cfg: Config
) -> Dict[str, np.ndarray]:
    """
    r_4h, done_4h, step_mask_4h
    - terminal reward at terminal step: survival_reward or death_reward
    - optional shaping from SOFA change across 4h step (requires cfg.shaping_alpha > 0)

    Key fix:
      - If mortality indicates death but deathtime missing, still assign death_reward,
        and place terminal step using episode_end_hours as best-effort.
    """
    _, t_4h = _fixed_time_grids(cfg)
    N = len(cohort)
    K = len(t_4h)

    mortality = cohort["mortality_in_hosp"].to_numpy(dtype=np.int32)
    death_hours = cohort["death_hours"].to_numpy(dtype=np.float32)
    end_hours = cohort["episode_end_hours"].to_numpy(dtype=np.float32)

    terminal_step = np.zeros((N,), dtype=np.int64)
    terminal_is_death = np.zeros((N,), dtype=np.int32)

    for i in range(N):
        is_death = (mortality[i] == 1)
        if is_death:
            tau = death_hours[i] if (not np.isnan(death_hours[i])) else end_hours[i]
            terminal_step[i] = compute_terminal_step(float(tau), cfg)
            terminal_is_death[i] = 1
        else:
            terminal_step[i] = compute_terminal_step(float(end_hours[i]), cfg)
            terminal_is_death[i] = 0

    step_mask = np.zeros((N, K), dtype=np.float32)
    done = np.zeros((N, K), dtype=np.float32)
    r = np.zeros((N, K), dtype=np.float32)

    for i in range(N):
        k = int(terminal_step[i])
        step_mask[i, :k+1] = 1.0
        done[i, k] = 1.0
        r[i, k] += cfg.death_reward if terminal_is_death[i] == 1 else cfg.survival_reward

    if cfg.shaping_alpha > 0 and (sofa is not None) and (not sofa.empty):
        sofa_1h = build_sofa_1h(cohort, sofa, cfg)  # (N, 73)
        for i in range(N):
            for k in range(K):
                if step_mask[i, k] == 0:
                    continue
                h0 = int(t_4h[k])
                h1 = min(cfg.horizon_hours, h0 + cfg.decision_dt_hours)
                s0 = sofa_1h[i, h0]
                s1 = sofa_1h[i, h1] if h1 <= cfg.horizon_hours else sofa_1h[i, -1]
                if np.isnan(s0) or np.isnan(s1):
                    continue
                shaping = -cfg.shaping_alpha * float(s1 - s0)
                shaping = float(np.clip(shaping, -cfg.shaping_clip, cfg.shaping_clip))
                r[i, k] += shaping

    return {"r_4h": r, "done_4h": done, "step_mask_4h": step_mask, "terminal_step": terminal_step}


# =============================================================================
# SOFA 1H (AUX / SHAPING ONLY)
# =============================================================================

def build_sofa_1h(cohort: pd.DataFrame, sofa: pd.DataFrame, cfg: Config) -> np.ndarray:
    """
    Best-effort SOFA alignment to 1h grid:
    - For each sofa row [starttime, endtime), fill hours in that range with sofa_24hours if present,
      else sum of components.
    Returns (N, 73) with NaN where unknown, and 0 beyond time_mask.
    """
    t_1h, _ = _fixed_time_grids(cfg)
    N = len(cohort)
    T = len(t_1h)

    stay_ids = cohort["stay_id"].astype(int).tolist()
    stay_to_i = {sid: i for i, sid in enumerate(stay_ids)}
    t0_map = cohort.set_index("stay_id")["t0"].to_dict()
    end_map = cohort.set_index("stay_id")["episode_end_hours"].to_dict()
    time_mask = np.zeros((N, T), dtype=np.float32)
    end_hours = cohort["episode_end_hours"].to_numpy(dtype=np.float32)
    for i in range(N):
        last_valid = int(min(cfg.horizon_hours, math.floor(float(end_hours[i]))))
        time_mask[i, :last_valid + 1] = 1.0

    S = np.full((N, T), np.nan, dtype=np.float32)

    if sofa.empty:
        return S

    s = sofa.copy()
    s = s[s["stay_id"].isin(stay_ids)].copy()
    s["starttime"] = s["starttime"].apply(_to_datetime_safe)
    s["endtime"] = s["endtime"].apply(_to_datetime_safe)
    s["endtime"] = s["endtime"].fillna(s["starttime"])

    def sofa_total(row) -> float:
        if "sofa_24hours" in row and not pd.isna(row["sofa_24hours"]):
            return _safe_float(row["sofa_24hours"])
        comp = 0.0
        for c in ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]:
            if c in row and not pd.isna(row[c]):
                comp += float(row[c])
        return comp

    for row in s.to_dict(orient="records"):
        sid = int(row["stay_id"])
        i = stay_to_i.get(sid)
        if i is None:
            continue
        t0 = t0_map.get(sid)
        if pd.isna(t0):
            continue
        st = row["starttime"]
        et = row["endtime"]
        if pd.isna(st) or pd.isna(et):
            continue
        # compute overlap with [t0, t0+horizon]
        clip_start = max(st, t0)
        clip_end = min(et, t0 + pd.to_timedelta(cfg.horizon_hours, unit="h"))
        if clip_end <= clip_start:
            continue
        h_start = int(max(0, math.floor((clip_start - t0).total_seconds() / 3600.0)))
        h_end = int(min(cfg.horizon_hours, math.ceil((clip_end - t0).total_seconds() / 3600.0)))
        val = float(sofa_total(row))
        for h in range(h_start, h_end + 1):
            if 0 <= h < T:
                S[i, h] = val

    # forward fill within time_mask
    for i in range(N):
        last = np.nan
        for t in range(T):
            if time_mask[i, t] == 0:
                S[i, t] = np.nan
                continue
            if not np.isnan(S[i, t]):
                last = S[i, t]
            else:
                if not np.isnan(last):
                    S[i, t] = last
    return S


# =============================================================================
# TEXT FUSION (RADIOLOGY + MICROBIOLOGY)
# =============================================================================

def _available_time(df: pd.DataFrame, chart_col: str, store_col: str) -> pd.Series:
    chart = df[chart_col] if chart_col in df.columns else pd.Series([pd.NaT]*len(df))
    store = df[store_col] if store_col in df.columns else pd.Series([pd.NaT]*len(df))
    return store.fillna(chart)

def build_text_tensors(
    cohort: pd.DataFrame,
    radiology: pd.DataFrame,
    micro: pd.DataFrame,
    cfg: Config
) -> Dict[str, np.ndarray]:
    """
    For each stay and each decision time t in {0,4,...,68}, produce:
      - e_rad: embedding of the most recent available radiology note up to t
      - e_micro: embedding of the most recent available micro event up to t
      - m_text: MNAR-style features:
          [rad_has_any, rad_count_cum, rad_hours_since_last,
           micro_has_any, micro_count_cum, micro_hours_since_last]

    Key fixes (top-tier correctness):
      1) hadm-level notes/events are mapped to stays by MERGE (supports multiple ICU stays per hadm_id)
      2) availability time = storetime if present else charttime
      3) strict leakage control: keep only events with 0 <= hours_from_t0 <= min(horizon, episode_end_hours)
    """
    _, t_4h = _fixed_time_grids(cfg)
    N = len(cohort)
    K = len(t_4h)

    embedder = create_text_embedder(cfg)
    # Update text_dim based on actual embedder (important for ClinicalBERT)
    actual_text_dim = embedder.text_dim
    if actual_text_dim != cfg.text_dim:
        print(f"[build_text_embeddings] Updating text_dim: {cfg.text_dim} -> {actual_text_dim}")

    stay_ids = cohort["stay_id"].astype(int).tolist()
    stay_to_i = {sid: i for i, sid in enumerate(stay_ids)}

    # outputs - use actual embedder dimension
    actual_text_dim = embedder.text_dim
    e_rad = np.zeros((N, K, actual_text_dim), dtype=np.float32)
    e_micro = np.zeros((N, K, actual_text_dim), dtype=np.float32)
    m_text = np.zeros((N, K, 6), dtype=np.float32)

    # ---------- helper ----------
    def avail_time(df: pd.DataFrame, chart_col: str = "charttime", store_col: str = "storetime") -> pd.Series:
        if store_col in df.columns and chart_col in df.columns:
            return df[store_col].fillna(df[chart_col])
        if store_col in df.columns:
            return df[store_col]
        return df.get(chart_col, pd.Series([pd.NaT] * len(df)))

    # cohort keys we need for per-stay alignment
    cohort_key = cohort[["hadm_id", "stay_id", "t0", "episode_end_hours"]].copy()
    cohort_key["stay_id"] = cohort_key["stay_id"].astype(int)
    cohort_key["hadm_id"] = cohort_key["hadm_id"].astype(int)

    # store per stay: list of (hours_from_t0, text)
    rad_notes: Dict[int, List[Tuple[float, str]]] = {sid: [] for sid in stay_ids}
    micro_events: Dict[int, List[Tuple[float, str]]] = {sid: [] for sid in stay_ids}

    # ---------- radiology ----------
    if radiology is not None and (not radiology.empty):
        rad = radiology.copy()

        for c in ["charttime", "storetime"]:
            if c in rad.columns:
                rad[c] = rad[c].apply(_to_datetime_safe)

        if "text" not in rad.columns:
            rad["text"] = ""

        rad["avail_time"] = avail_time(rad, "charttime", "storetime").apply(_to_datetime_safe)
        rad["text"] = rad["text"].apply(lambda x: _clean_text(_clip_text(x, cfg.text_max_chars), cfg.text_lower))

        # map hadm-level notes to stays via merge (supports multi-stay per hadm)
        if "hadm_id" not in rad.columns:
            # if user provides stay-level radiology, try to recover hadm_id from cohort
            if "stay_id" in rad.columns:
                stay_to_hadm = cohort.set_index("stay_id")["hadm_id"].to_dict()
                rad["hadm_id"] = rad["stay_id"].map(stay_to_hadm)
        rad = rad[rad["hadm_id"].notna()].copy()
        rad["hadm_id"] = rad["hadm_id"].astype(int)

        rad = rad.merge(cohort_key, on="hadm_id", how="inner")

        rad["hours_from_t0"] = (rad["avail_time"] - rad["t0"]).dt.total_seconds() / 3600.0
        rad = rad[rad["hours_from_t0"].notna()].copy()

        # strict clamp by episode_end_hours (prevents post-death/discharge info entering state)
        rad["max_h"] = np.minimum(float(cfg.horizon_hours), rad["episode_end_hours"].astype(float))
        rad = rad[(rad["hours_from_t0"] >= 0) & (rad["hours_from_t0"] <= rad["max_h"])].copy()

        for row in rad[["stay_id", "hours_from_t0", "text"]].itertuples(index=False):
            sid = int(row.stay_id)
            if sid in rad_notes:
                rad_notes[sid].append((float(row.hours_from_t0), str(row.text)))

        for sid in rad_notes:
            rad_notes[sid].sort(key=lambda x: x[0])

    # ---------- micro ----------
    if micro is not None and (not micro.empty):
        mi = micro.copy()
        for c in ["charttime", "storetime"]:
            if c in mi.columns:
                mi[c] = mi[c].apply(_to_datetime_safe)

        mi["avail_time"] = avail_time(mi, "charttime", "storetime").apply(_to_datetime_safe)

        # build concise text per micro event (you may drop ab_name if you worry about “non-action clinical decisions” confounding)
        def build_micro_text(r: pd.Series) -> str:
            parts = []
            for c in ["spec_type_desc", "test_name", "org_name", "interpretation", "comments"]:
                if c in r and not pd.isna(r[c]):
                    parts.append(str(r[c]))
            return _clean_text(_clip_text(" | ".join(parts), cfg.text_max_chars), cfg.text_lower)

        mi["text"] = mi.apply(build_micro_text, axis=1)

        mi = mi[mi["hadm_id"].notna()].copy()
        mi["hadm_id"] = mi["hadm_id"].astype(int)

        mi = mi.merge(cohort_key, on="hadm_id", how="inner")
        mi["hours_from_t0"] = (mi["avail_time"] - mi["t0"]).dt.total_seconds() / 3600.0
        mi = mi[mi["hours_from_t0"].notna()].copy()

        mi["max_h"] = np.minimum(float(cfg.horizon_hours), mi["episode_end_hours"].astype(float))
        mi = mi[(mi["hours_from_t0"] >= 0) & (mi["hours_from_t0"] <= mi["max_h"])].copy()

        for row in mi[["stay_id", "hours_from_t0", "text"]].itertuples(index=False):
            sid = int(row.stay_id)
            if sid in micro_events:
                micro_events[sid].append((float(row.hours_from_t0), str(row.text)))

        for sid in micro_events:
            micro_events[sid].sort(key=lambda x: x[0])

    # ---------- embed (deduplicate exact strings) ----------
    def embed_events(events_by_stay: Dict[int, List[Tuple[float, str]]]) -> Dict[int, List[Tuple[float, np.ndarray]]]:
        uniq: Dict[str, int] = {}
        texts: List[str] = []
        for sid, evs in events_by_stay.items():
            for _, txt in evs:
                if txt not in uniq:
                    uniq[txt] = len(texts)
                    texts.append(txt)
        embs = embedder.encode(texts) if texts else np.zeros((0, embedder.text_dim), dtype=np.float32)

        out: Dict[int, List[Tuple[float, np.ndarray]]] = {sid: [] for sid in events_by_stay.keys()}
        for sid, evs in events_by_stay.items():
            for t, txt in evs:
                idx = uniq.get(txt, None)
                if idx is not None:
                    out[sid].append((t, embs[idx]))
            out[sid].sort(key=lambda x: x[0])
        return out

    rad_emb_events = embed_events(rad_notes)
    mic_emb_events = embed_events(micro_events)

    # ---------- fill tensors ----------
    for sid in stay_ids:
        i = stay_to_i[sid]

        rad_evs = rad_emb_events.get(sid, [])
        rad_times = [t for t, _ in rad_evs]

        mic_evs = mic_emb_events.get(sid, [])
        mic_times = [t for t, _ in mic_evs]

        for kk, t_dec in enumerate(t_4h):
            t_dec = float(t_dec)

            # radiology
            ridx = np.searchsorted(rad_times, t_dec, side="right") - 1
            if ridx >= 0:
                e_rad[i, kk] = rad_evs[ridx][1]
                m_text[i, kk, 0] = 1.0
                m_text[i, kk, 1] = float(ridx + 1)
                m_text[i, kk, 2] = float(t_dec - rad_times[ridx])
            else:
                m_text[i, kk, 0] = 0.0
                m_text[i, kk, 1] = 0.0
                m_text[i, kk, 2] = float(cfg.horizon_hours + 1)

            # micro
            midx = np.searchsorted(mic_times, t_dec, side="right") - 1
            if midx >= 0:
                e_micro[i, kk] = mic_evs[midx][1]
                m_text[i, kk, 3] = 1.0
                m_text[i, kk, 4] = float(midx + 1)
                m_text[i, kk, 5] = float(t_dec - mic_times[midx])
            else:
                m_text[i, kk, 3] = 0.0
                m_text[i, kk, 4] = 0.0
                m_text[i, kk, 5] = float(cfg.horizon_hours + 1)

    m_text[:, :, 2] = np.clip(m_text[:, :, 2], 0.0, float(cfg.horizon_hours + 1))
    m_text[:, :, 5] = np.clip(m_text[:, :, 5], 0.0, float(cfg.horizon_hours + 1))

    return {
        "e_rad": e_rad, 
        "e_micro": e_micro, 
        "m_text": m_text, 
        "t_4h": t_4h,
        "text_dim": embedder.text_dim,  # Pass actual dimension for metadata
        "text_backend": cfg.text_backend,
    }


# =============================================================================
# STATIC FEATURES
# =============================================================================

def build_static_features(cohort: pd.DataFrame, charlson: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Static features (minimal, leakage-safe):
      - age (anchor_age)
      - gender (M=1, F=0, else 0.5)
      - charlson_index
    """
    df = cohort.merge(charlson[["stay_id", "charlson_index"]], on="stay_id", how="left")
    age = df["anchor_age"].astype(float).fillna(df["anchor_age"].median()).to_numpy()
    gender_raw = df["gender"].astype(str).fillna("UNK").to_numpy()
    gender = np.array([1.0 if g.upper() == "M" else 0.0 if g.upper() == "F" else 0.5 for g in gender_raw], dtype=np.float32)
    charl = df["charlson_index"].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
    X = np.stack([age.astype(np.float32), gender, charl], axis=1).astype(np.float32)
    names = ["age", "gender_male", "charlson_index"]
    return X, names


# =============================================================================
# SANITY CHECKS
# =============================================================================

def run_sanity_checks(dataset: Dict[str, Any], cohort: pd.DataFrame, cfg: Config) -> None:
    np.random.seed(cfg.random_seed)
    N = len(cohort)
    n = min(cfg.sanity_check_n, N)
    idx = np.random.choice(np.arange(N), size=n, replace=False)

    Y = dataset["Y"]
    mask = dataset["mask"]
    time_mask = dataset["time_mask"]
    delta = dataset["delta"]
    a_4h = dataset["a_4h"]
    r_4h = dataset["r_4h"]
    done_4h = dataset["done_4h"]
    step_mask_4h = dataset["step_mask_4h"]

    assert Y.shape[0] == N and mask.shape == Y.shape and delta.shape == Y.shape, "Shape mismatch in obs tensors."
    assert time_mask.shape[:2] == Y.shape[:2], "Shape mismatch in time_mask."
    assert a_4h.shape == (N, 18), "a_4h shape mismatch."
    assert r_4h.shape == (N, 18) and done_4h.shape == (N, 18), "reward/done shape mismatch."

    # Check that beyond time_mask, mask is zero and Y is finite (we set to 0)
    for i in idx:
        invalid = (time_mask[i] == 0)
        if invalid.any():
            assert np.all(mask[i, invalid, :] == 0), "mask not zero beyond episode window."
            assert np.all(np.isfinite(Y[i, invalid, :])), "Y has non-finite beyond episode window."

    # Done should happen exactly once per trajectory within step_mask
    for i in idx:
        d = done_4h[i]
        assert d.sum() == 1.0, f"done_4h should have exactly one terminal 1; got {d.sum()}."
        term = int(np.argmax(d))
        assert step_mask_4h[i, term] == 1.0, "terminal step must be within step_mask."
        assert np.all(step_mask_4h[i, term+1:] == 0.0), "step_mask should be 0 after terminal."
        # reward nonzero only at terminal unless shaping on
        if cfg.shaping_alpha == 0.0:
            nz = np.where(np.abs(r_4h[i]) > 1e-6)[0]
            assert len(nz) == 1 and nz[0] == term, "reward should be terminal-only when shaping disabled."

    print(f"  Sanity checks passed on {n} random trajectories.")

    a_flat = a_4h.reshape(-1)
    counts = np.bincount(a_flat, minlength=9).astype(np.float64)
    props = counts / max(1.0, counts.sum())
    print("  [Stats] Action distribution over all 4h steps:")
    for a_id in range(9):
        print(f"    action={a_id}: {props[a_id]*100:.2f}% (n={int(counts[a_id])})")

    feat_names = dataset.get("feature_names", None)
    valid = time_mask[:, :, None]  # (N,T,1)
    obs_count = (mask * valid).sum(axis=(0, 1))  # (D,)
    tot_count = valid.sum(axis=(0, 1)).astype(np.float64)  # (1,)
    tot_count = float(tot_count) if np.ndim(tot_count) == 0 else float(tot_count[0])
    miss_rate = 1.0 - (obs_count / max(1.0, tot_count))

    print("  [Stats] Missing rate per feature (within episode window):")
    for d in range(mask.shape[2]):
        name = feat_names[d] if (feat_names is not None and d < len(feat_names)) else f"feat_{d}"
        print(f"    {name}: {miss_rate[d]*100:.2f}%")


# =============================================================================
# SAVE
# =============================================================================

def save_outputs(output_dir: str, arrays: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    _ensure_dir(output_dir)
    # Save arrays as .npy (fast load, memmap-friendly)
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            np.save(os.path.join(output_dir, f"{k}.npy"), v)
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved arrays + metadata to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MIMIC-IV Sepsis Offline RL + Controlled-SDE preprocessing (v2)")
    parser.add_argument("--sepsis_data_dir", type=str, default=Config.sepsis_data_dir)
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)
    parser.add_argument("--cache_dir", type=str, default=Config.cache_dir)

    parser.add_argument("--no_bigquery", action="store_true", help="Disable BigQuery and use local CSV fallbacks where possible.")
    parser.add_argument("--gcp_billing_project", type=str, default=Config.gcp_billing_project)
    parser.add_argument("--mimic_project", type=str, default=Config.mimic_project)
    parser.add_argument("--mimic_version_prefix", type=str, default=Config.mimic_version_prefix)
    parser.add_argument("--mimic_note_dataset", type=str, default=Config.mimic_note_dataset)

    parser.add_argument("--text_dim", type=int, default=Config.text_dim,
                        help="Text embedding dimension (768 for clinicalbert, 256 for hashing)")
    parser.add_argument("--text_backend", type=str, default=Config.text_backend,
                        choices=["clinicalbert", "hashing"],
                        help="Text embedding backend: 'clinicalbert' (recommended) or 'hashing'")
    parser.add_argument("--text_bert_model", type=str, default=Config.text_bert_model,
                        help="HuggingFace model name for ClinicalBERT backend")
    parser.add_argument("--text_bert_device", type=str, default=Config.text_bert_device,
                        choices=["cuda", "cpu"], help="Device for BERT inference")
    parser.add_argument("--fluids_high_cutoff_ml", type=float, default=float("nan"))
    parser.add_argument("--vaso_high_cutoff_ne", type=float, default=float("nan"))

    parser.add_argument("--survival_reward", type=float, default=Config.survival_reward)
    parser.add_argument("--death_reward", type=float, default=Config.death_reward)
    parser.add_argument("--shaping_alpha", type=float, default=Config.shaping_alpha)

    parser.add_argument("--run_sanity_checks", action="store_true", help="Run internal sanity checks on random trajectories.")
    
    parser.add_argument("--bq_cache_only", action="store_true", help="Only read BigQuery cache files; never hit BigQuery.")
    parser.add_argument("--force_refresh_cache", action="store_true", help="Ignore cache and re-query BigQuery.")
    parser.add_argument("--no_print_cache_hits", action="store_true", help="Disable cache hit logging.")

    args = parser.parse_args()

    cfg = Config(
        sepsis_data_dir=args.sepsis_data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_bigquery=not args.no_bigquery,

        # NEW:
        bq_cache_only=bool(args.bq_cache_only),
        force_refresh_cache=bool(args.force_refresh_cache),
        print_cache_hits=not bool(args.no_print_cache_hits),

        gcp_billing_project=args.gcp_billing_project,
        mimic_project=args.mimic_project,
        mimic_version_prefix=args.mimic_version_prefix,
        mimic_note_dataset=args.mimic_note_dataset,
        text_dim=args.text_dim,
        text_backend=args.text_backend,
        text_bert_model=args.text_bert_model,
        text_bert_device=args.text_bert_device,
        survival_reward=args.survival_reward,
        death_reward=args.death_reward,
        shaping_alpha=args.shaping_alpha,
    )

    # optional fixed cutoffs
    if not np.isnan(args.fluids_high_cutoff_ml):
        cfg.fluids_high_cutoff_ml = float(args.fluids_high_cutoff_ml)
    if not np.isnan(args.vaso_high_cutoff_ne):
        cfg.vaso_high_cutoff_ne = float(args.vaso_high_cutoff_ne)

    np.random.seed(cfg.random_seed)

    print("=" * 90)
    print("MIMIC-IV SEPSIS OFFLINE RL + CONTROLLED-SDE PREPROCESSING (v2)")
    print("=" * 90)

    # Load local inputs
    local = load_local_inputs(cfg)
    cohort = build_master_cohort(local, cfg)
    print(f"[Cohort] N={len(cohort):,} stays | Mortality={cohort['mortality_in_hosp'].mean()*100:.2f}%")

    # Charlson
    charlson = compute_charlson_features(local["diagnoses"], cohort)

    # BigQuery fetch
    vitals = pd.DataFrame()
    labs = pd.DataFrame()
    sofa = pd.DataFrame()
    fluids = pd.DataFrame()
    ne = pd.DataFrame()
    rad = pd.DataFrame()
    micro = pd.DataFrame()

    if cfg.use_bigquery:
        fetcher = BigQueryFetcher(cfg)
        try:
            print("[BigQuery] Fetching vitals/labs/sofa/treatments/text ...")
            vitals = fetch_vitals_bq(fetcher, cohort, cfg)
            labs = fetch_labs_bq(fetcher, cohort, cfg)
            sofa = fetch_sofa_bq(fetcher, cohort, cfg)
            fluids = fetch_fluids_bq(fetcher, cohort, cfg)
            ne = fetch_ne_equiv_bq(fetcher, cohort, cfg)
            rad = fetch_radiology_bq(fetcher, cohort, cfg)
            micro = fetch_micro_bq(fetcher, cohort, cfg)
        except BigQueryUnavailable as e:
            print(f"  [WARN] BigQuery unavailable: {e}")
            cfg.use_bigquery = False
        except Exception as e:
            print(f"  [WARN] BigQuery fetch failed: {e}")
            cfg.use_bigquery = False

    # Fallbacks if no BigQuery
    if not cfg.use_bigquery:
        print("[Fallback] Using local CSVs where possible (coverage may be limited).")
        vitals = local.get("vitals_local", pd.DataFrame())
        labs = local.get("labs_first_48h", pd.DataFrame())
        rad = local.get("radiology_notes_local", pd.DataFrame())
        # treatments/micro/sofa not available locally in this project snapshot
        sofa = pd.DataFrame()
        fluids = pd.DataFrame()
        ne = pd.DataFrame()
        micro = pd.DataFrame()

        # standardize local time columns
        if not vitals.empty and "charttime" in vitals.columns:
            vitals["charttime"] = vitals["charttime"].apply(_to_datetime_safe)
        if not labs.empty and "charttime" in labs.columns:
            labs["charttime"] = labs["charttime"].apply(_to_datetime_safe)
        if not rad.empty:
            for c in ["charttime", "storetime"]:
                if c in rad.columns:
                    rad[c] = rad[c].apply(_to_datetime_safe)

    # Build observation tensors
    print("[Build] 1h observation tensors...")
    obs = build_obs_tensors(cohort, vitals, labs, cfg)
    ff = forward_fill_and_deltas(obs["Y_raw"], obs["mask"], obs["time_mask"])

    # Build continuous exposures and discretize actions
    print("[Build] 4h actions from fluids + NE-equivalent dose...")
    exposures = build_continuous_exposures(cohort, fluids, ne, cfg)
    disc = discretize_actions(exposures["fluids_ml_4h"], exposures["vaso_ne_4h"], cfg)

    # Reward/done
    print("[Build] rewards/done (terminal mortality reward; optional SOFA shaping)...")
    rd = build_reward_done(cohort, sofa, cfg)

    # Text tensors
    print("[Build] text embeddings (radiology + microbiology) at 4h decision times...")
    text = build_text_tensors(cohort, rad, micro, cfg)

    # Static
    X_static, static_names = build_static_features(cohort, charlson)

    # Package arrays
    arrays: Dict[str, Any] = {
        "stay_id": cohort["stay_id"].astype(np.int64).to_numpy(),
        "hadm_id": cohort["hadm_id"].astype(np.int64).to_numpy(),
        "subject_id": cohort["subject_id"].astype(np.int64).to_numpy(),
        "X_static": X_static.astype(np.float32),

        "t_1h": obs["t_1h"].astype(np.int32),
        "Y": ff["Y"].astype(np.float32),
        "mask": obs["mask"].astype(np.float32),
        "delta": ff["delta"].astype(np.float32),
        "time_mask": obs["time_mask"].astype(np.float32),

        "t_4h": exposures["t_4h"].astype(np.int32),
        "fluids_ml_4h": exposures["fluids_ml_4h"].astype(np.float32),
        "vaso_ne_4h": exposures["vaso_ne_4h"].astype(np.float32),
        "a_4h": disc["a_4h"].astype(np.int64),
        "a_1h": disc["a_1h"].astype(np.int64),

        "r_4h": rd["r_4h"].astype(np.float32),
        "done_4h": rd["done_4h"].astype(np.float32),
        "step_mask_4h": rd["step_mask_4h"].astype(np.float32),

        "e_rad": text["e_rad"].astype(np.float32),
        "e_micro": text["e_micro"].astype(np.float32),
        "m_text": text["m_text"].astype(np.float32),
    }

    metadata: Dict[str, Any] = {
        "config": asdict(cfg),
        "n_stays": int(len(cohort)),
        "horizon_hours": int(cfg.horizon_hours),
        "obs_dt_hours": int(cfg.obs_dt_hours),
        "decision_dt_hours": int(cfg.decision_dt_hours),
        "obs_feature_names": obs["feature_names"],
        "static_feature_names": static_names,
        "text_dim": int(text.get("text_dim", cfg.text_dim)),  # Use actual dim from embedder
        "text_backend": text.get("text_backend", cfg.text_backend),
        "text_backend": cfg.text_backend,
        "fluids_cutoff_ml": float(disc["fluids_cutoff_ml"]),
        "vaso_cutoff_ne": float(disc["vaso_cutoff_ne"]),
        "action_space": {
            "fluids_levels": 3,
            "vaso_levels": 3,
            "n_actions": 9,
            "action_id_formula": "action = fluids_level*3 + vaso_level",
        },
        "reward": {
            "survival_reward": cfg.survival_reward,
            "death_reward": cfg.death_reward,
            "shaping_alpha": cfg.shaping_alpha,
        },
        "outcomes": {
            "mortality_in_hosp_rate": float(cohort["mortality_in_hosp"].mean()),
        },
        "notes": [
            "State leakage control: radiology/micro embeddings only use storetime (or charttime if storetime null) <= decision time.",
            "Treatments require BigQuery; without BigQuery, actions may be all-zeros.",
        ],
    }

    # Sanity checks
    if args.run_sanity_checks:
        dataset_for_check = {
            "Y": arrays["Y"],
            "mask": arrays["mask"],
            "delta": arrays["delta"],
            "time_mask": arrays["time_mask"],
            "a_4h": arrays["a_4h"],
            "r_4h": arrays["r_4h"],
            "done_4h": arrays["done_4h"],
            "step_mask_4h": arrays["step_mask_4h"],
            "feature_names": obs["feature_names"],  # NEW
        }
        run_sanity_checks(dataset_for_check, cohort, cfg)

    # Save
    _ensure_dir(cfg.output_dir)
    save_outputs(cfg.output_dir, arrays, metadata)

    print("\nDONE.")
    print(f"Output directory: {cfg.output_dir}")
    print("Saved files include: *.npy arrays and metadata.json")


if __name__ == "__main__":
    main()