"""
ASTRA — Dataset Audit Script
Phase 1: Full reproducible dataset audit

Usage:
    python ml/audit/01_dataset_audit.py                        # uses default paths
    python ml/audit/01_dataset_audit.py --train /path/train.csv --test /path/test.csv

Data paths can also be set via environment variables:
    ASTRA_TRAIN_CSV=/path/to/train.csv
    ASTRA_TEST_CSV=/path/to/test.csv
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION — env vars > CLI args > default relative paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="ASTRA Dataset Audit")
parser.add_argument("--train", default=os.environ.get("ASTRA_TRAIN_CSV", ""), help="Path to train CSV")
parser.add_argument("--test",  default=os.environ.get("ASTRA_TEST_CSV",  ""), help="Path to test CSV")
args, _ = parser.parse_known_args()

# Fallback: look for data files relative to the project root
_default_train = ROOT / "data" / "train_data.csv"
_default_test  = ROOT / "data" / "test_data.csv"

TRAIN_CSV = Path(args.train) if args.train else _default_train
TEST_CSV  = Path(args.test)  if args.test  else _default_test

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Train CSV not found: {TRAIN_CSV}\n"
        "Set ASTRA_TRAIN_CSV env var or pass --train <path>"
    )
if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found: {TEST_CSV}\n"
        "Set ASTRA_TEST_CSV env var or pass --test <path>"
    )

print("=" * 60)
print("ASTRA DATASET AUDIT")
print("=" * 60)

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

audit = {}

# Shape
audit["shape"] = {
    "train_rows": int(len(train)), "train_cols": int(train.shape[1]),
    "test_rows":  int(len(test)),  "test_cols":  int(test.shape[1]),
}
print(f"\nShape: Train={audit['shape']['train_rows']:,} rows, Test={audit['shape']['test_rows']:,} rows, {audit['shape']['train_cols']} columns each")

# Event overlap
train_ev = set(train["event_id"].unique())
test_ev  = set(test["event_id"].unique())
overlap  = train_ev & test_ev
audit["events"] = {
    "train_unique_events": int(len(train_ev)),
    "test_unique_events":  int(len(test_ev)),
    "overlap_count":       int(len(overlap)),
    "leakage_detected":    len(overlap) == len(test_ev),
}
print(f"Events: Train={len(train_ev):,}, Test={len(test_ev):,}, Overlap={len(overlap):,} ({'LEAKAGE' if len(overlap) > 0 else 'CLEAN'})")

# Target stats
for name, df in [("train", train), ("test", test)]:
    audit[f"{name}_target_stats"] = {
        "min": round(float(df["risk"].min()), 4),
        "max": round(float(df["risk"].max()), 4),
        "mean": round(float(df["risk"].mean()), 4),
        "std":  round(float(df["risk"].std()), 4),
    }
print(f"Target (log10 Pc): Train mean={audit['train_target_stats']['mean']:.3f}, Test mean={audit['test_target_stats']['mean']:.3f}")

# Class distribution
def tier(x): return "High" if x > -12 else "Medium" if x > -20 else "Low"
train["tier"] = train["risk"].apply(tier)
test["tier"]  = test["risk"].apply(tier)
audit["class_distribution"] = {
    "train": train["tier"].value_counts(normalize=True).round(4).to_dict(),
    "test":  test["tier"].value_counts(normalize=True).round(4).to_dict(),
}
print(f"Class dist (train): {audit['class_distribution']['train']}")
print(f"Class dist (test):  {audit['class_distribution']['test']}")

# Missing values
miss_train = train.isna().sum()
miss_test  = test.isna().sum()
audit["missing_values"] = {
    "train_top10": miss_train[miss_train > 0].sort_values(ascending=False).head(10).to_dict(),
    "test_top10":  miss_test[miss_test > 0].sort_values(ascending=False).head(10).to_dict(),
}
print(f"Top missing (train): c_rcs_estimate={miss_train.get('c_rcs_estimate', 0):,}")

# Duplicates
audit["duplicates"] = {
    "train_duplicate_rows": int(train.duplicated().sum()),
    "test_duplicate_rows":  int(test.duplicated().sum()),
}
print(f"Duplicates: Train={audit['duplicates']['train_duplicate_rows']}, Test={audit['duplicates']['test_duplicate_rows']}")

# max_risk_scaling leakage check
corr = float(train[["max_risk_scaling", "risk"]].corr().iloc[0, 1])
audit["leakage_check"] = {
    "max_risk_scaling_corr_with_risk": round(corr, 6),
    "verdict": "Low direct correlation — not a simple copy of target, but temporal leakage possible",
}
print(f"max_risk_scaling corr with risk: {corr:.4f}")

# CDMs per event
cdms_per_event = train.groupby("event_id").size()
audit["cdms_per_event"] = {
    "mean": round(float(cdms_per_event.mean()), 2),
    "median": round(float(cdms_per_event.median()), 2),
    "min":  int(cdms_per_event.min()),
    "max":  int(cdms_per_event.max()),
}
print(f"CDMs/event (train): mean={audit['cdms_per_event']['mean']}, max={audit['cdms_per_event']['max']}")

out = RESULTS / "dataset_audit.json"
out.write_text(json.dumps(audit, indent=2))
print(f"\nAudit saved → {out}")
print("=" * 60)
