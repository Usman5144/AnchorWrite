from quality_conditions_helper import find_input, get_conditions

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


BASE=str(Path(__file__).resolve().parents[1])

INPUT=None

OUT=f"{BASE}/results/reproduced/quality/quality_analysis"

os.makedirs(OUT,exist_ok=True)


conditions = get_conditions()


# ==========================================================
# Load data
# ==========================================================

data={}

for c in conditions:

    path=find_input(c)

    df=pd.read_csv(path)

    data[c]=df

    print(c,len(df))


# ==========================================================
# MiniLM semantic similarity
# ==========================================================

print("\nLoading MiniLM...")

model_path=os.environ["ANCHORWRITE_MINILM_MODEL"]

embedder=SentenceTransformer(
    model_path,
    device="cuda"
)


original=data["original"]["text"].astype(str).tolist()

print("Encoding original")

orig_emb=embedder.encode(
    original,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True
)


semantic=[]


for c in conditions:

    if c=="original":

        sims=np.ones(len(original))

    else:

        texts=data[c]["text"].astype(str).tolist()

        print("Encoding",c)

        emb=embedder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Align samples for conditions with missing rows
        n=min(len(orig_emb),len(emb))

        sims=np.sum(
            orig_emb[:n]*emb[:n],
            axis=1
        )


    semantic.append(
        pd.DataFrame([{
            "condition":c,
            "mean_similarity":float(np.mean(sims)),
            "median_similarity":float(np.median(sims)),
            "std_similarity":float(np.std(sims)),
            "min_similarity":float(np.min(sims))
        }])
    )


semantic_df=pd.concat(
    semantic,
    ignore_index=True
)


semantic_df.to_csv(
    f"{OUT}/semantic_similarity.csv",
    index=False
)


# ==========================================================
# Diversity
# ==========================================================

def distinct(texts,n):

    total=0
    unique=set()

    for t in texts:

        tokens=t.split()

        total+=max(len(tokens)-n+1,0)

        for i in range(len(tokens)-n+1):

            unique.add(
                " ".join(tokens[i:i+n])
            )

    return len(unique)/total


div=[]


for c in conditions:

    texts=data[c]["text"].astype(str).tolist()

    div.append({

        "condition":c,

        "distinct_1":distinct(texts,1),

        "distinct_2":distinct(texts,2),

        "avg_length":

        np.mean(
            [len(x.split()) for x in texts]
        )

    })


div_df=pd.DataFrame(div)


div_df.to_csv(
    f"{OUT}/diversity.csv",
    index=False
)



# ==========================================================
# Summary
# ==========================================================

summary=semantic_df.merge(
    div_df,
    on="condition"
)


summary.to_csv(
    f"{OUT}/quality_summary.csv",
    index=False
)


print()
print("==============================")
print("QUALITY ANALYSIS COMPLETE")
print("==============================")

print(summary)

print()

print("Saved:")
print(OUT)

