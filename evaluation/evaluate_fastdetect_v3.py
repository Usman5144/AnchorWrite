from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data/detector_scores/main/fastdetect"
OUT_DIR = ROOT / "results/reproduced/detectors/fastdetect"


conditions=[
    "original",
    "L1",
    "L2",
    "L3",
    "L4"
]


human=pd.read_csv(
    f"{DIR}/human_fastdetect_scores.csv"
)


rows=[]


for condition in conditions:

    ai=pd.read_csv(
        f"{DIR}/{condition}_fastdetect_scores.csv"
    )


    labels=[1]*len(ai)+[0]*len(human)

    scores=list(ai.fastdetect_score)+list(
        human.fastdetect_score
    )


    auc1=roc_auc_score(
        labels,
        scores
    )

    auc2=roc_auc_score(
        labels,
        [-x for x in scores]
    )


    auc=max(
        auc1,
        auc2
    )


    final_scores = (
        scores
        if auc1>=auc2
        else [-x for x in scores]
    )


    rows.append(
        {
            "condition":condition,
            "AUROC":auc,
            "AUPRC":
                average_precision_score(
                    labels,
                    final_scores
                ),
            "mean_ai":
                ai.fastdetect_score.mean(),
            "mean_human":
                human.fastdetect_score.mean()
        }
    )


out=pd.DataFrame(rows)

print(out)

OUT_DIR.mkdir(parents=True, exist_ok=True)

out.to_csv(
    OUT_DIR / "fastdetect_metrics.csv",
    index=False
)

print("Saved")
