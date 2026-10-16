from quality_conditions_helper import find_input, get_conditions

import os
from pathlib import Path
import re
import gc
import torch
import pandas as pd
from bert_score import score as bert_score


BASE = str(Path(__file__).resolve().parents[1])

INPUT = f"{BASE}/data/evaluation/main"

OUT_DIR = f"{BASE}/results/reproduced/quality/semantic"

MODEL_PATH = os.environ["ANCHORWRITE_BERTSCORE_MODEL"]

SUMMARY_OUT = f"{OUT_DIR}/bertscore.csv"

SAMPLE_OUT = f"{OUT_DIR}/bertscore_per_sample.csv"


CONDITIONS = get_conditions()


os.makedirs(OUT_DIR, exist_ok=True)


def canonical_sample_id(sample_id):
    """
    Normalize sample IDs across generators.

    Examples:
        original_social_sciences_0
        social_sciences_0
        social_sciences_moonshot_150_0

    ->  social_sciences_0
    """

    x = str(sample_id).strip()

    if x.startswith("original_"):
        x = x[len("original_"):]


    parts = x.split("_")

    # Remove generator/model suffix before final index
    # e.g. essay_chatgpt_150_0 -> essay_0
    if len(parts) >= 3 and parts[-2] == "150":
        x = "_".join(
            parts[:-2] + [parts[-1]]
        )


    # chatgpt/moonshot variants
    x = x.replace("_chatgpt", "")
    x = x.replace("_moonshot", "")

    return x


def load_condition(condition):
    path = find_input(condition)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"sample_id", "domain", "text"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{condition}: missing columns {sorted(missing)}"
        )

    if len(df) < 100:
        raise ValueError(
            f"{condition}: too few valid samples ({len(df)})"
        )

    print(
        f"{condition}: aligned samples = {len(df)}"
    )

    df = df.copy()

    df["canonical_id"] = df["sample_id"].map(
        canonical_sample_id
    )

    df["domain"] = (
        df["domain"]
        .astype(str)
        .str.replace("_chatgpt_150", "", regex=False)
        .str.replace("_moonshot_150", "", regex=False)
        .str.replace("_150", "", regex=False)
    )

    if df[["domain", "canonical_id"]].duplicated().any():
        bad = df[
            df[["domain", "canonical_id"]].duplicated(
                keep=False
            )
        ][["domain", "sample_id", "canonical_id"]]

        raise ValueError(
            f"{condition}: duplicate canonical IDs:\n{bad.head(20)}"
        )

    return df


print("=" * 70)
print("BERTScore QUALITY EVALUATION")
print("=" * 70)

print("Model:")
print(MODEL_PATH)

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"RoBERTa-large model directory not found: {MODEL_PATH}"
    )

if not os.path.exists(
    os.path.join(MODEL_PATH, "config.json")
):
    raise FileNotFoundError(
        f"config.json missing from {MODEL_PATH}"
    )

has_safetensors = os.path.exists(
    os.path.join(MODEL_PATH, "model.safetensors")
)

has_bin = os.path.exists(
    os.path.join(MODEL_PATH, "pytorch_model.bin")
)

if not (has_safetensors or has_bin):
    raise FileNotFoundError(
        "No model.safetensors or pytorch_model.bin found."
    )


print()
print("Loading and validating inputs...")


original = load_condition("original")

original_ref = original[
    ["domain", "canonical_id", "text"]
].copy()

original_ref = original_ref.rename(
    columns={"text": "reference_text"}
)

original_ref["_order"] = range(len(original_ref))


sample_frames = []

summary_records = []


# Original compared with itself.
original_scores = pd.DataFrame({
    "condition": "original",
    "domain": original_ref["domain"],
    "sample_id": original_ref["canonical_id"],
    "bertscore_precision": 1.0,
    "bertscore_recall": 1.0,
    "bertscore_f1": 1.0,
})

sample_frames.append(original_scores)

summary_records.append({
    "condition": "original",
    "mean_bertscore_precision": 1.0,
    "median_bertscore_precision": 1.0,
    "std_bertscore_precision": 0.0,
    "mean_bertscore_recall": 1.0,
    "median_bertscore_recall": 1.0,
    "std_bertscore_recall": 0.0,
    "mean_bertscore_f1": 1.0,
    "median_bertscore_f1": 1.0,
    "std_bertscore_f1": 0.0,
    "min_bertscore_f1": 1.0,
    "samples": len(original_ref),
})


for condition in CONDITIONS[1:]:

    print()
    print("=" * 70)
    print("Processing:", condition)
    print("=" * 70)

    candidate = load_condition(condition)

    candidate = candidate[
        ["domain", "canonical_id", "text"]
    ].copy()

    candidate = candidate.rename(
        columns={"text": "candidate_text"}
    )


    paired = original_ref.merge(
        candidate,
        on=["domain", "canonical_id"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    paired = paired.sort_values(
        "_order"
    ).reset_index(drop=True)


    if len(paired) < 100:
        raise ValueError(
            f"{condition}: too few aligned pairs ({len(paired)})"
        )

    print(
        f"{condition}: final aligned pairs = {len(paired)}"
    )


    original_keys = set(
        zip(
            original_ref["domain"],
            original_ref["canonical_id"]
        )
    )

    candidate_keys = set(
        zip(
            candidate["domain"],
            candidate["canonical_id"]
        )
    )

    if original_keys != candidate_keys:

        missing = original_keys - candidate_keys
        extra = candidate_keys - original_keys

        if extra:
            raise ValueError(
                f"{condition}: unexpected extra sample keys. "
                f"extra={len(extra)}"
            )

        if missing:
            print(
                f"{condition}: missing {len(missing)} samples "
                f"(failed attack generations), continuing with aligned pairs"
            )


    print("Aligned pairs:", len(paired))


    candidates = paired[
        "candidate_text"
    ].astype(str).tolist()

    references = paired[
        "reference_text"
    ].astype(str).tolist()


    # For standard roberta-large BERTScore,
    # the recommended layer is 17.
    #
    # We deliberately use raw BERTScore without
    # rescaling-with-baseline so that the pipeline
    # works completely offline with the frozen
    # local evaluator.
    P, R, F1 = bert_score(
        cands=candidates,
        refs=references,
        model_type=MODEL_PATH,
        num_layers=17,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
        idf=False,
        rescale_with_baseline=False,
    )


    p = P.detach().cpu().numpy()
    r = R.detach().cpu().numpy()
    f1 = F1.detach().cpu().numpy()


    sample_df = pd.DataFrame({
        "condition": condition,
        "domain": paired["domain"],
        "sample_id": paired["canonical_id"],
        "bertscore_precision": p,
        "bertscore_recall": r,
        "bertscore_f1": f1,
    })

    sample_frames.append(sample_df)


    p_s = pd.Series(p)
    r_s = pd.Series(r)
    f_s = pd.Series(f1)


    summary_records.append({
        "condition": condition,

        "mean_bertscore_precision":
            float(p_s.mean()),

        "median_bertscore_precision":
            float(p_s.median()),

        "std_bertscore_precision":
            float(p_s.std()),

        "mean_bertscore_recall":
            float(r_s.mean()),

        "median_bertscore_recall":
            float(r_s.median()),

        "std_bertscore_recall":
            float(r_s.std()),

        "mean_bertscore_f1":
            float(f_s.mean()),

        "median_bertscore_f1":
            float(f_s.median()),

        "std_bertscore_f1":
            float(f_s.std()),

        "min_bertscore_f1":
            float(f_s.min()),

        "samples":
            len(f_s),
    })


    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


summary = pd.DataFrame(
    summary_records
)

order_map = {
    condition: i
    for i, condition in enumerate(CONDITIONS)
}

summary["_order"] = summary[
    "condition"
].map(order_map)

summary = summary.sort_values(
    "_order"
).drop(columns="_order")


per_sample = pd.concat(
    sample_frames,
    ignore_index=True
)

per_sample["_order"] = per_sample[
    "condition"
].map(order_map)

per_sample = per_sample.sort_values(
    ["_order", "domain", "sample_id"]
).drop(columns="_order")


summary.to_csv(
    SUMMARY_OUT,
    index=False
)

per_sample.to_csv(
    SAMPLE_OUT,
    index=False
)


print()
print("=" * 70)
print("BERTScore QUALITY COMPLETE")
print("=" * 70)

print(summary.to_string(index=False))

print()
print("Saved:")
print(SUMMARY_OUT)
print(SAMPLE_OUT)
