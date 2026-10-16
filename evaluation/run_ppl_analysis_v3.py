from quality_conditions_helper import find_input, get_conditions

import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE=str(Path(__file__).resolve().parents[1])

INPUT=None

OUT=f"{BASE}/results/reproduced/quality/quality_analysis"

os.makedirs(OUT, exist_ok=True)


MODEL_PATH=os.environ["ANCHORWRITE_GPT_NEO_MODEL"]


device="cuda" if torch.cuda.is_available() else "cpu"


print("Loading GPT-Neo...")

tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)

model=AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()


def calculate_ppl(text):

    enc=tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids=enc.input_ids.to(model.device)

    with torch.no_grad():

        outputs=model(
            input_ids,
            labels=input_ids
        )

        loss=outputs.loss

    ppl=torch.exp(loss)

    return float(ppl.cpu())


records=[]


for condition in get_conditions():

    path=find_input(condition)

    df=pd.read_csv(path)

    print()
    print("Processing:", condition)
    print("Samples:",len(df))

    values=[]

    for text in tqdm(df["text"]):

        values.append(
            calculate_ppl(str(text))
        )


    records.append(
        {
            "condition":condition,
            "mean_PPL":sum(values)/len(values),
            "median_PPL":float(pd.Series(values).median()),
            "std_PPL":float(pd.Series(values).std()),
            "min_PPL":min(values),
            "max_PPL":max(values)
        }
    )


result=pd.DataFrame(records)


result.to_csv(
    f"{OUT}/perplexity_analysis.csv",
    index=False
)


print()
print("==============================")
print("PPL ANALYSIS COMPLETE")
print("==============================")

print(result)

print()
print("Saved:")
print(f"{OUT}/perplexity_analysis.csv")
