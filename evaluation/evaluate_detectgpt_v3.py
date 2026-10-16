import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = str(
    ROOT / "data/detector_scores/main/detectgpt"
)

OUT_DIR = (
    ROOT / "results/reproduced/detectors/detectgpt"
)

files = {
    "original": "original_detectgpt_scores.csv",

    "hmgc_L1": "hmgc_L1_detectgpt_scores.csv",
    "hmgc_L2": "hmgc_L2_detectgpt_scores.csv",
    "hmgc_L3": "hmgc_L3_detectgpt_scores.csv",
    "hmgc_L4": "hmgc_L4_detectgpt_scores.csv",
}

results = []


# Human scores
human_file = str(
    ROOT / "data/detector_scores/baselines/fastdetect/human_fastdetect_scores.csv"
)

# DetectGPT needs human negative scores.
# Use original human condition from detector inputs if available.
human_input = pd.read_csv(
    ROOT / "data/evaluation/main/human.csv"
)

# temporary: load DetectGPT human scores if generated later
human_scores_file = os.path.join(
    RESULT_DIR,
    "human_detectgpt_scores.csv"
)

if not os.path.exists(human_scores_file):
    raise FileNotFoundError(
        "Human DetectGPT scores missing. Run DetectGPT on human.csv first."
    )

human_df = pd.read_csv(human_scores_file)

human_df = human_df.dropna(
    subset=["detectgpt_score"]
)

human_scores = human_df["detectgpt_score"].values


for condition, filename in files.items():

    df = pd.read_csv(
        os.path.join(RESULT_DIR, filename)
    )

    before = len(df)

    df = df.dropna(
        subset=["detectgpt_score"]
    )

    after = len(df)

    print(f"{condition}: removed {before-after} NaN scores")

    ai_scores = df["detectgpt_score"].values

    scores = np.concatenate(
        [human_scores, ai_scores]
    )

    labels = np.concatenate(
        [
            np.zeros(len(human_scores)),
            np.ones(len(ai_scores))
        ]
    )

    auc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    pred = (scores > np.median(scores)).astype(int)

    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred)

    results.append(
        {
            "condition": condition,
            "AUROC": auc,
            "AUPRC": auprc,
            "Accuracy": acc,
            "F1": f1,
            "mean_AI": np.mean(ai_scores),
            "mean_Human": np.mean(human_scores)
        }
    )


out = pd.DataFrame(results)

print(out.round(4))

OUT_DIR.mkdir(parents=True, exist_ok=True)

out.to_csv(
    OUT_DIR / "detectgpt_metrics.csv",
    index=False
)

print("\nSaved")
