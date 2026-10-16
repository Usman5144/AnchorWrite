
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEARCH_DIRS = [
    ROOT / "data" / "evaluation" / "main",
    ROOT / "data" / "evaluation" / "baselines",
]


def find_input(condition):

    for d in SEARCH_DIRS:
        p = d / f"{condition}.csv"
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        f"Missing quality input: {condition}"
    )



def get_conditions():

    priority = [
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
        "hmgc_L4",
    ]

    return [
        x for x in priority
        if any(
            (d / f"{x}.csv").exists()
            for d in SEARCH_DIRS
        )
    ]
