import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)


ROOT = Path(__file__).resolve().parents[1]
DIR = str(ROOT / "data/detector_scores/baselines/roberta")
OUT_DIR = ROOT / "results/reproduced/detectors/roberta"


rows=[]


human_df=pd.read_csv(
    os.path.join(
        DIR,
        "human_roberta_scores.csv"
    )
)


# Human labels
human_scores = human_df["roberta_ai_probability"].values
human_labels = [0]*len(human_scores)


for cond in [
    "original",

    "qwen_rewrite_L1",
    "qwen_rewrite_L2",
    "qwen_rewrite_L3",
    "qwen_rewrite_L4",

    "qwen_humanize_L1",
    "qwen_humanize_L2",
    "qwen_humanize_L3",
    "qwen_humanize_L4",

    "pegasus_L1",
    "pegasus_L2",
    "pegasus_L3",
    "pegasus_L4",

    "backtranslation_L1",
    "backtranslation_L2",
    "backtranslation_L3",
    "backtranslation_L4",

    "dipper_L1",
    "dipper_L2",
    "dipper_L3",
    "dipper_L4",

    "gramguard_L1",
    "gramguard_L2",
    "gramguard_L3",
    "gramguard_L4",

    "hmgc_L1",
    "hmgc_L2",
    "hmgc_L3",
    "hmgc_L4"
]:

    df=pd.read_csv(
        os.path.join(
            DIR,
            cond+"_roberta_scores.csv"
        )
    )


    # AI scores = fake probability
    ai_scores = df["roberta_ai_probability"].values

    scores = list(ai_scores) + list(human_scores)

    labels = (
        [1]*len(ai_scores)
        +
        human_labels
    )


    pred = [
        1 if x>=0.5 else 0
        for x in scores
    ]


    rows.append({

        "condition":cond,

        "AUROC":
            roc_auc_score(
                labels,
                scores
            ),

        "AUPRC":
            average_precision_score(
                labels,
                scores
            ),

        "Accuracy":
            accuracy_score(
                labels,
                pred
            ),

        "F1":
            f1_score(
                labels,
                pred
            )

    })


out=pd.DataFrame(rows)


print(out)


OUT_DIR.mkdir(parents=True, exist_ok=True)

out.to_csv(
    OUT_DIR / "roberta_detection_metrics.csv",
    index=False
)


print("\nSaved roberta_detection_metrics.csv")
