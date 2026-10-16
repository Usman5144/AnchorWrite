from quality_conditions_helper import find_input, get_conditions

import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE=str(Path(__file__).resolve().parents[1])


def canonical_sample_id(sample_id):
    x = str(sample_id).strip()

    for prefix in [
        "original_",
        "dipper_",
        "pegasus_",
        "backtranslation_",
        "qwen_rewrite_",
        "qwen_humanize_",
    ]:
        if x.startswith(prefix):
            x = x[len(prefix):]

    return x


INPUT=f"{BASE}/data/evaluation/main"

OUT=f"{BASE}/results/reproduced/quality/semantic/nli.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


MODEL_PATH=os.environ["ANCHORWRITE_NLI_MODEL"]


DEVICE="cuda" if torch.cuda.is_available() else "cpu"


conditions = get_conditions()


print("==============================")
print("Loading NLI model")
print("==============================")


tokenizer=AutoTokenizer.from_pretrained(
    MODEL_PATH
)


model=AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()


# Actual mapping for DeBERTa-v3-base-mnli-fever-anli
# 0 entailment
# 1 neutral
# 2 contradiction


def nli_score(premise, hypothesis):

    inputs=tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt"
    )

    inputs={
        k:v.to(DEVICE)
        for k,v in inputs.items()
    }


    with torch.no_grad():

        logits=model(**inputs).logits

        probs=torch.softmax(
            logits,
            dim=-1
        )[0]


    return {
        "entailment":float(probs[0]),
        "neutral":float(probs[1]),
        "contradiction":float(probs[2])
    }



original=pd.read_csv(
    f"{INPUT}/original.csv"
)


original_texts=original["text"].tolist()


records=[]


for condition in conditions:

    print()
    print("Processing:",condition)


    humanized=pd.read_csv(
        find_input(condition)
    )



        # Normalize IDs for different generation pipelines

    original["canonical_id"] = original["sample_id"].map(
        canonical_sample_id
    )

    humanized["canonical_id"] = humanized["sample_id"].map(
        canonical_sample_id
    )


    original["domain"] = (
        original["domain"]
        .astype(str)
    )

    humanized["domain"] = (
        humanized["domain"]
        .astype(str)
        .str.replace("_chatgpt_150","",regex=False)
        .str.replace("_moonshot_150","",regex=False)
        .str.replace("_150","",regex=False)
    )


    original_map = dict(
        zip(
            zip(original["domain"], original["canonical_id"]),
            original["text"]
        )
    )


    human_map = dict(
        zip(
            zip(humanized["domain"], humanized["canonical_id"]),
            humanized["text"]
        )
    )


    common_keys = sorted(
        set(original_map.keys()) &
        set(human_map.keys())
    )


    print(
        f"{condition}: aligned samples = {len(common_keys)}"
    )


    if len(common_keys) < 100:
        raise ValueError(
            f"{condition}: too few aligned samples ({len(common_keys)})"
        )


    original_texts = [
        original_map[k]
        for k in common_keys
    ]

    human_texts = [
        human_map[k]
        for k in common_keys
    ]


    entailments=[]
    contradictions=[]
    neutrals=[]


    for o,h in tqdm(
        zip(original_texts,human_texts),
        total=len(original_texts)
    ):


        # original -> humanized
        s1=nli_score(
            o,
            h
        )


        # humanized -> original
        s2=nli_score(
            h,
            o
        )


        entailments.append(
            (s1["entailment"]+
             s2["entailment"])/2
        )


        contradictions.append(
            (s1["contradiction"]+
             s2["contradiction"])/2
        )


        neutrals.append(
            (s1["neutral"]+
             s2["neutral"])/2
        )


    records.append({

        "condition":condition,

        "mean_bidirectional_entailment":
            sum(entailments)/len(entailments),

        "mean_bidirectional_contradiction":
            sum(contradictions)/len(contradictions),

        "mean_bidirectional_neutral":
            sum(neutrals)/len(neutrals),

        "std_entailment":
            pd.Series(entailments).std(),

        "std_contradiction":
            pd.Series(contradictions).std(),

        "std_neutral":
            pd.Series(neutrals).std(),

        "samples":
            len(entailments)

    })


df=pd.DataFrame(records)


df.to_csv(
    OUT,
    index=False
)


print()
print("==============================")
print("NLI QUALITY COMPLETE")
print("==============================")

print(df)


print()
print("Saved:")
print(OUT)
