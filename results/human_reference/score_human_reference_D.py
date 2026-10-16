#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import sys

import kenlm
import numpy as np
import pandas as pd


# ============================================================
# Frozen paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MAIN_IMPL = ROOT / "code/scripts/gramguard_v3.py"

KENLM_2 = ROOT / "calibration/umbc_2gram.binary"
KENLM_4 = ROOT / "calibration/umbc_4gram.trie.binary"

CALIBRATION = (
    ROOT / "calibration/umbc_human_null_antique_features_10000.csv"
)

INPUT_DIR = ROOT / "evaluation_v3/detector_inputs"
OUT_DIR = ROOT / "analysis_human_reference"

CONDITIONS = {
    "Held-out Human": INPUT_DIR / "human.csv",
    "Original AI": INPUT_DIR / "original.csv",
    "L1": INPUT_DIR / "L1.csv",
    "L2": INPUT_DIR / "L2.csv",
    "L3": INPUT_DIR / "L3.csv",
    "L4": INPUT_DIR / "L4.csv",
}

REL_WINDOW = 0.25
MIN_NEIGHBORS = 100

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260924


# ============================================================
# Import exact frozen implementation
# ============================================================

spec = importlib.util.spec_from_file_location(
    "anchorwrite_frozen_impl",
    MAIN_IMPL,
)

if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import {MAIN_IMPL}")

impl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = impl
spec.loader.exec_module(impl)


# ============================================================
# Utilities
# ============================================================

def bootstrap_mean_ci(values, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    means = np.empty(reps, dtype=float)

    for b in range(reps):
        sample = rng.choice(x, size=len(x), replace=True)
        means[b] = sample.mean()

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def score_text(text, model2, model4, calibration):
    text = str(text)

    if not impl.clean_text(text):
        raise ValueError("Encountered empty text.")

    length = int(impl.word_count(text))

    h2 = float(impl.kenlm_cross_entropy(model2, text))
    h4 = float(impl.kenlm_cross_entropy(model4, text))

    if not np.isfinite(h2) or not np.isfinite(h4):
        raise RuntimeError(
            f"Non-finite KenLM value: H2={h2}, H4={h4}"
        )

    a24 = float(
        (h2 - h4) /
        max(h2, 1e-12)
    )

    stats = impl.human_null_stats(
        length=length,
        a24=a24,
        calibration=calibration,
        rel_window=REL_WINDOW,
        min_neighbors=MIN_NEIGHBORS,
    )

    return {
        "word_count": length,
        "H2": h2,
        "H4": h4,
        "A24": a24,
        "reference_center": float(stats["human_null_center"]),
        "reference_scale": float(stats["human_null_scale"]),
        "reference_neighbors": int(stats["human_null_neighbors"]),
        "signed_z": float(stats["human_null_signed_z"]),
        "D": float(stats["human_null_distance"]),
    }


# ============================================================
# Validate inputs
# ============================================================

print("============================================================")
print("AnchorWrite held-out human-reference validation")
print("============================================================")

for path in [MAIN_IMPL, KENLM_2, KENLM_4, CALIBRATION]:
    if not path.exists():
        raise FileNotFoundError(path)

input_frames = {}

for condition, path in CONDITIONS.items():

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"sample_id", "domain", "text"}

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{condition}: missing columns {sorted(missing)}"
        )

    if len(df) != 900:
        raise RuntimeError(
            f"{condition}: expected 900 rows, found {len(df)}"
        )

    if df["text"].isna().any():
        raise RuntimeError(
            f"{condition}: contains missing text."
        )

    counts = df["domain"].value_counts().sort_index()

    if len(counts) != 6 or not (counts == 150).all():
        raise RuntimeError(
            f"{condition}: unexpected domain counts:\n{counts}"
        )

    input_frames[condition] = df

    print(
        f"{condition:16s}: "
        f"{len(df)} docs | "
        f"domains={len(counts)}"
    )


# ============================================================
# Load frozen reference assets
# ============================================================

print()
print("[LOAD] frozen human-reference calibration")
calibration = impl.load_human_null_calibration(CALIBRATION)

print()
print(f"[LOAD] 2-gram: {KENLM_2}")
model2 = kenlm.Model(str(KENLM_2))

print(f"[LOAD] 4-gram: {KENLM_4}")
model4 = kenlm.Model(str(KENLM_4))


# ============================================================
# Score every text directly
# ============================================================

all_rows = []

for condition, df in input_frames.items():

    print()
    print("============================================================")
    print(f"SCORING: {condition}")
    print("============================================================")

    condition_rows = []

    for j, (_, row) in enumerate(df.iterrows(), start=1):

        result = score_text(
            row["text"],
            model2,
            model4,
            calibration,
        )

        record = {
            "condition": condition,
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            **result,
        }

        condition_rows.append(record)

        if j % 100 == 0 or j == len(df):
            print(f"{condition}: {j}/{len(df)}")

    scored = pd.DataFrame(condition_rows)

    # --------------------------------------------------------
    # Internal reproduction check against already stored
    # final D for rewritten documents.
    #
    # We DO NOT use those stored values for analysis.
    # They are only an implementation consistency check.
    # --------------------------------------------------------

    if condition in {"L1", "L2", "L3", "L4"}:

        source = df.reset_index(drop=True)

        if "human_null_after" in source.columns:

            stored = pd.to_numeric(
                source["human_null_after"],
                errors="coerce",
            )

            mask = stored.notna()

            if mask.any():

                recomputed = scored.loc[
                    mask.to_numpy(),
                    "D"
                ].to_numpy(dtype=float)

                old = stored.loc[mask].to_numpy(dtype=float)

                diff = np.abs(recomputed - old)

                print(
                    f"[CHECK] {condition}: "
                    f"stored={len(diff)}, "
                    f"mean_abs_diff={diff.mean():.10f}, "
                    f"max_abs_diff={diff.max():.10f}"
                )

                if diff.max() > 1e-5:
                    raise RuntimeError(
                        f"{condition}: recomputed D does not reproduce "
                        f"stored final D. max_abs_diff={diff.max()}"
                    )

    all_rows.extend(condition_rows)


scores = pd.DataFrame(all_rows)

if len(scores) != 5400:
    raise RuntimeError(
        f"Expected 5400 scored documents, found {len(scores)}"
    )

scores_path = OUT_DIR / "human_reference_D_scores.csv"
scores.to_csv(scores_path, index=False)


# ============================================================
# TABLE A — aggregate
# ============================================================

aggregate_rows = []

condition_order = [
    "Held-out Human",
    "Original AI",
    "L1",
    "L2",
    "L3",
    "L4",
]

for i, condition in enumerate(condition_order):

    sub = scores.loc[
        scores["condition"] == condition,
        "D"
    ].to_numpy(dtype=float)

    lo, hi = bootstrap_mean_ci(
        sub,
        seed=BOOTSTRAP_SEED + i,
    )

    aggregate_rows.append(
        {
            "Condition": condition,
            "n": len(sub),
            "Mean_D": np.mean(sub),
            "Median_D": np.median(sub),
            "Std_D": np.std(sub, ddof=1),
            "Q25_D": np.percentile(sub, 25),
            "Q75_D": np.percentile(sub, 75),
            "Mean_D_CI95_Low": lo,
            "Mean_D_CI95_High": hi,
        }
    )


aggregate = pd.DataFrame(aggregate_rows)

aggregate_path = (
    OUT_DIR /
    "human_reference_D_aggregate.csv"
)

aggregate.to_csv(
    aggregate_path,
    index=False,
)


# ============================================================
# TABLE B — domain-level
# ============================================================

domain_rows = []

domains = sorted(
    scores["domain"]
    .dropna()
    .unique()
)

for domain in domains:

    def domain_mean(condition):
        x = scores.loc[
            (scores["condition"] == condition) &
            (scores["domain"] == domain),
            "D",
        ]

        if len(x) != 150:
            raise RuntimeError(
                f"{domain}/{condition}: expected 150 rows, "
                f"found {len(x)}"
            )

        return float(x.mean())

    human_d = domain_mean("Held-out Human")
    original_d = domain_mean("Original AI")
    l4_d = domain_mean("L4")

    domain_rows.append(
        {
            "Domain": domain,
            "Human_Mean_D": human_d,
            "Original_AI_Mean_D": original_d,

            # Positive = original AI lies farther from
            # the human-reference center than held-out humans.
            "Original_minus_Human": original_d - human_d,

            "L4_Mean_D": l4_d,
        }
    )


domain_table = pd.DataFrame(domain_rows)

domain_path = (
    OUT_DIR /
    "human_reference_D_domains.csv"
)

domain_table.to_csv(
    domain_path,
    index=False,
)


# ============================================================
# Independent Human-vs-Original diagnostic
# ============================================================

human = scores.loc[
    scores["condition"] == "Held-out Human",
    "D",
].to_numpy(dtype=float)

original = scores.loc[
    scores["condition"] == "Original AI",
    "D",
].to_numpy(dtype=float)

summary = pd.DataFrame(
    [{
        "Human_n": len(human),
        "Original_AI_n": len(original),
        "Human_Mean_D": human.mean(),
        "Original_AI_Mean_D": original.mean(),
        "Original_minus_Human_Mean_D":
            original.mean() - human.mean(),
        "Human_Median_D": np.median(human),
        "Original_AI_Median_D": np.median(original),
    }]
)

summary_path = (
    OUT_DIR /
    "human_vs_original_D_validation.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Print compact results
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print()
print("============================================================")
print("TABLE A — AGGREGATE")
print("============================================================")
print(
    aggregate.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("============================================================")
print("TABLE B — DOMAIN")
print("============================================================")
print(
    domain_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("============================================================")
print("HUMAN vs ORIGINAL")
print("============================================================")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Saved:")
print(scores_path)
print(aggregate_path)
print(domain_path)
print(summary_path)
