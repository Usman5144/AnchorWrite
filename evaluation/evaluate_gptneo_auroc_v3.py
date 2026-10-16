import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DIR = str(ROOT / "data/detector_scores/main/gptneo")
OUT_DIR = ROOT / "results/reproduced/detectors/gptneo"


human=pd.read_csv(
    os.path.join(
        DIR,
        "human_gptneo_scores.csv"
    )
)

human = human.dropna(
    subset=[
        "loglikelihood",
        "logrank",
        "entropy"
    ]
)


rows=[]


metrics=[
    "loglikelihood",
    "logrank",
    "entropy"
]


# direction testing
def best_auc(labels, scores):

    a1=roc_auc_score(
        labels,
        scores
    )

    a2=roc_auc_score(
        labels,
        [-x for x in scores]
    )

    return max(a1,a2)



for condition in [
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

    ai=pd.read_csv(
        os.path.join(
            DIR,
            condition+"_gptneo_scores.csv"
        )
    )

    ai = ai.dropna(
        subset=[
            "loglikelihood",
            "logrank",
            "entropy"
        ]
    )


    labels=[1]*len(ai)+[0]*len(human)


    row={"condition":condition}


    for m in metrics:

        scores=list(ai[m])+list(human[m])

        row[m+"_AUROC"]=best_auc(
            labels,
            scores
        )


    rows.append(row)



out=pd.DataFrame(rows)

print(out)


OUT_DIR.mkdir(parents=True, exist_ok=True)

out.to_csv(
    OUT_DIR / "gptneo_detector_metrics.csv",
    index=False
)


print("Saved")
