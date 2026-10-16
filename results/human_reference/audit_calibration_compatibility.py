#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import sys

import kenlm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

IMPL = ROOT / "code/scripts/gramguard_v3.py"
CAL = ROOT / "calibration/umbc_human_null_antique_features_10000.csv"
K2 = ROOT / "calibration/umbc_2gram.binary"
K4 = ROOT / "calibration/umbc_4gram.trie.binary"

OUT = ROOT / "analysis_human_reference/calibration_compatibility_50.csv"

# Exact current implementation
spec = importlib.util.spec_from_file_location("anchorwrite_impl", IMPL)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

cal = pd.read_csv(CAL)

# Fixed deterministic spread across the entire calibration table
idx = np.linspace(0, len(cal)-1, 50, dtype=int)
sub = cal.iloc[idx].copy()

print("Loading frozen KenLM models...")
m2 = kenlm.Model(str(K2))
m4 = kenlm.Model(str(K4))

rows = []

for j, (orig_idx, r) in enumerate(sub.iterrows(), start=1):

    text = str(r["text"])

    stored_h2 = float(r["H2"])
    stored_h4 = float(r["H4"])
    stored_a24 = (stored_h2 - stored_h4) / max(stored_h2, 1e-12)

    current_h2 = float(mod.kenlm_cross_entropy(m2, text))
    current_h4 = float(mod.kenlm_cross_entropy(m4, text))
    current_a24 = (
        current_h2 - current_h4
    ) / max(current_h2, 1e-12)

    rows.append({
        "calib_row": int(orig_idx),
        "stored_num_words": r["num_words"],
        "current_word_count": mod.word_count(text),

        "stored_H2": stored_h2,
        "current_H2": current_h2,
        "absdiff_H2": abs(stored_h2-current_h2),

        "stored_H4": stored_h4,
        "current_H4": current_h4,
        "absdiff_H4": abs(stored_h4-current_h4),

        "stored_A24": stored_a24,
        "current_A24": current_a24,
        "absdiff_A24": abs(stored_a24-current_a24),
    })

    print(
        f"{j:02d}/50 "
        f"row={orig_idx:5d} "
        f"A24 stored={stored_a24:.8f} "
        f"current={current_a24:.8f} "
        f"diff={abs(stored_a24-current_a24):.10f}"
    )

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("\n============================================================")
print("CALIBRATION COMPATIBILITY SUMMARY")
print("============================================================")

for col in ["absdiff_H2", "absdiff_H4", "absdiff_A24"]:
    print(
        f"{col:14s} "
        f"mean={out[col].mean():.12f} "
        f"median={out[col].median():.12f} "
        f"max={out[col].max():.12f}"
    )

print("\nWord-count exact matches:",
      int((out["stored_num_words"] == out["current_word_count"]).sum()),
      "/ 50")

print("\nStored A24:")
print(
    f"mean={out.stored_A24.mean():.6f} "
    f"median={out.stored_A24.median():.6f}"
)

print("Current A24:")
print(
    f"mean={out.current_A24.mean():.6f} "
    f"median={out.current_A24.median():.6f}"
)

print("\nSaved:", OUT)
