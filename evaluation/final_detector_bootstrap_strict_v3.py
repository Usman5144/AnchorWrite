#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

MASTER_DIR = (
    ROOT
    / "results"
    / "paper"
)

MASTER_FILE = (
    MASTER_DIR
    / "12_detector_AUROC_master_standardized.csv"
)

OUT = (
    ROOT
    / "results"
    / "reproduced"
    / "bootstrap_strict"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

N_BOOT = 2000
SEED = 20260917

# Raw score must reproduce canonical AUROC before bootstrap is allowed.
TOL = 5e-4

LAYERS = [
    "L1",
    "L2",
    "L3",
    "L4",
]

METHODS = [
    "GramGuard",
    "Qwen Humanize",
    "Pegasus",
]


# =====================================================================
# LOAD CANONICAL MASTER
# =====================================================================

master = pd.read_csv(
    MASTER_FILE
)


def canonical_auc(
    detector,
    method,
    layer,
):

    q = master[
        (master["Detector"] == detector)
        &
        (master["Method"] == method)
        &
        (master["Layer"] == layer)
    ]

    if q.empty:
        return np.nan

    return float(
        q.iloc[0]["AUROC"]
    )


# =====================================================================
# HELPERS
# =====================================================================

def canonical_key(df):

    if "sample_id" not in df.columns:
        raise RuntimeError(
            "sample_id missing"
        )

    if "domain" not in df.columns:
        raise RuntimeError(
            "domain missing"
        )

    sid = (
        df["sample_id"]
        .astype(str)
    )

    number = sid.str.extract(
        r"(\d+)(?!.*\d)",
        expand=False
    )

    if number.isna().any():
        raise RuntimeError(
            "Could not extract numeric sample index."
        )

    domain = (
        df["domain"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    key = (
        domain
        +
        "::"
        +
        number.astype(str)
    )

    if key.duplicated().any():
        raise RuntimeError(
            "Duplicate canonical key."
        )

    return key


def prepare(
    path,
    score_col,
    sign,
):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    if score_col not in df.columns:
        raise RuntimeError(
            f"{path}: {score_col} missing"
        )

    df = df.copy()

    df["key"] = canonical_key(
        df
    )

    df["score"] = (
        sign
        *
        pd.to_numeric(
            df[score_col],
            errors="coerce"
        )
    )

    df = (
        df[
            [
                "key",
                "score",
            ]
        ]
        .dropna()
    )

    if df["key"].duplicated().any():
        raise RuntimeError(
            f"{path}: duplicate key"
        )

    return df


def auc_from_frames(
    human,
    ai,
):

    h = human["score"].to_numpy()
    a = ai["score"].to_numpy()

    y = np.r_[
        np.ones(len(a)),
        np.zeros(len(h)),
    ]

    s = np.r_[
        a,
        h,
    ]

    return roc_auc_score(
        y,
        s
    )


def verify(
    detector,
    method,
    layer,
    human,
    ai,
):

    raw = auc_from_frames(
        human,
        ai,
    )

    canon = canonical_auc(
        detector,
        method,
        layer,
    )

    if not np.isfinite(canon):

        raise RuntimeError(
            f"{detector} {method} {layer}: "
            "canonical AUROC missing"
        )

    diff = abs(
        raw
        -
        canon
    )

    if diff > TOL:

        raise RuntimeError(
            f"{detector} {method} {layer}: "
            f"RAW AUROC {raw:.6f} does not reproduce "
            f"canonical {canon:.6f}; diff={diff:.6f}"
        )

    return raw, canon, diff


def bootstrap_delta(
    human,
    original,
    condition,
    n_boot,
    seed,
):

    ai = (
        original
        .rename(
            columns={
                "score":
                    "original_score"
            }
        )
        .merge(
            condition.rename(
                columns={
                    "score":
                        "condition_score"
                }
            ),
            on="key",
            how="inner",
            validate="one_to_one",
        )
    )

    h = human["score"].to_numpy()

    o = ai["original_score"].to_numpy()
    c = ai["condition_score"].to_numpy()

    n_ai = len(ai)
    n_h = len(h)

    if n_ai != 900:
        raise RuntimeError(
            f"Expected 900 paired AI rows; got {n_ai}"
        )

    if n_h != 900:
        raise RuntimeError(
            f"Expected 900 human rows; got {n_h}"
        )

    rng = np.random.default_rng(
        seed
    )

    bo = np.empty(
        n_boot
    )

    bc = np.empty(
        n_boot
    )

    bd = np.empty(
        n_boot
    )

    for i in range(
        n_boot
    ):

        ia = rng.integers(
            0,
            n_ai,
            size=n_ai,
        )

        ih = rng.integers(
            0,
            n_h,
            size=n_h,
        )

        y = np.r_[
            np.ones(n_ai),
            np.zeros(n_h),
        ]

        hs = h[
            ih
        ]

        ao = roc_auc_score(
            y,
            np.r_[
                o[ia],
                hs,
            ]
        )

        ac = roc_auc_score(
            y,
            np.r_[
                c[ia],
                hs,
            ]
        )

        bo[i] = ao
        bc[i] = ac
        bd[i] = (
            ao
            -
            ac
        )

    y = np.r_[
        np.ones(n_ai),
        np.zeros(n_h),
    ]

    point_original = roc_auc_score(
        y,
        np.r_[
            o,
            h,
        ]
    )

    point_condition = roc_auc_score(
        y,
        np.r_[
            c,
            h,
        ]
    )

    return {

        "N_AI":
            n_ai,

        "N_human":
            n_h,

        "Original_AUROC":
            point_original,

        "Condition_AUROC":
            point_condition,

        "Delta_AUROC":
            point_original
            -
            point_condition,

        "Condition_CI_low":
            np.quantile(
                bc,
                0.025,
            ),

        "Condition_CI_high":
            np.quantile(
                bc,
                0.975,
            ),

        "Delta_CI_low":
            np.quantile(
                bd,
                0.025,
            ),

        "Delta_CI_high":
            np.quantile(
                bd,
                0.975,
            ),

        "P_delta_gt_0":
            np.mean(
                bd > 0
            ),
    }


def bootstrap_difference(
    human,
    baseline,
    gramguard,
    n_boot,
    seed,
):

    ai = (
        baseline
        .rename(
            columns={
                "score":
                    "baseline_score"
            }
        )
        .merge(
            gramguard.rename(
                columns={
                    "score":
                        "gramguard_score"
                }
            ),
            on="key",
            how="inner",
            validate="one_to_one",
        )
    )

    h = human["score"].to_numpy()

    b = ai["baseline_score"].to_numpy()
    g = ai["gramguard_score"].to_numpy()

    n_ai = len(ai)
    n_h = len(h)

    if n_ai != 900:
        raise RuntimeError(
            f"Expected 900 paired AI rows; got {n_ai}"
        )

    rng = np.random.default_rng(
        seed
    )

    diffs = np.empty(
        n_boot
    )

    for i in range(
        n_boot
    ):

        ia = rng.integers(
            0,
            n_ai,
            size=n_ai,
        )

        ih = rng.integers(
            0,
            n_h,
            size=n_h,
        )

        y = np.r_[
            np.ones(n_ai),
            np.zeros(n_h),
        ]

        hs = h[
            ih
        ]

        auc_b = roc_auc_score(
            y,
            np.r_[
                b[ia],
                hs,
            ]
        )

        auc_g = roc_auc_score(
            y,
            np.r_[
                g[ia],
                hs,
            ]
        )

        # Positive => baseline AUROC > GramGuard AUROC.
        diffs[i] = (
            auc_b
            -
            auc_g
        )

    y = np.r_[
        np.ones(n_ai),
        np.zeros(n_h),
    ]

    point_b = roc_auc_score(
        y,
        np.r_[
            b,
            h,
        ]
    )

    point_g = roc_auc_score(
        y,
        np.r_[
            g,
            h,
        ]
    )

    return {

        "Baseline_AUROC":
            point_b,

        "GramGuard_AUROC":
            point_g,

        "Difference_Baseline_minus_GramGuard":
            point_b
            -
            point_g,

        "Difference_CI_low":
            np.quantile(
                diffs,
                0.025,
            ),

        "Difference_CI_high":
            np.quantile(
                diffs,
                0.975,
            ),

        "P_Baseline_gt_GramGuard":
            np.mean(
                diffs > 0
            ),
    }


# =====================================================================
# EXPLICIT FILE CONFIGURATION
# =====================================================================

BASE = (
    ROOT
    / "data"
    / "detector_scores"
    / "baselines"
)

V3 = (
    ROOT
    / "data"
    / "detector_scores"
    / "main"
)


CONFIG = {

    "FastDetect": {

        "dir":
            BASE
            / "fastdetect",

        "score_col":
            "fastdetect_score",

        "sign":
            +1.0,

        "human":
            "human_fastdetect_scores.csv",

        "original":
            "original_fastdetect_scores.csv",

        "pattern":
            "{condition}_fastdetect_scores.csv",
    },


    "RoBERTa": {

        "dir":
            BASE
            / "roberta",

        "score_col":
            "roberta_ai_probability",

        # Audited globally:
        # direct = 0.202055
        # negated = 0.797945
        "sign":
            +1.0,

        "human":
            "human_roberta_scores.csv",

        "original":
            "original_roberta_scores.csv",

        "pattern":
            "{condition}_roberta_scores.csv",
    },


    "GPTNeo-LogRank": {

        "dir":
            V3
            / "gptneo",

        "score_col":
            "logrank",

        "sign":
            -1.0,

        "human":
            "human_gptneo_scores.csv",

        "original":
            "original_gptneo_scores.csv",

        "pattern":
            "{condition}_gptneo_scores.csv",
    },
}


def condition_name(
    method,
    layer,
):

    if method == "GramGuard":
        return (
            f"gramguard_{layer}"
        )

    if method == "Qwen Humanize":
        return (
            f"qwen_humanize_{layer}"
        )

    if method == "Pegasus":
        return (
            f"pegasus_{layer}"
        )

    raise ValueError(
        method
    )


# =====================================================================
# RUN STANDARD RAW-SCORE DETECTORS
# =====================================================================

results = []
matched_results = []
audit = []

seed_counter = 0


for detector, cfg in CONFIG.items():

    print()
    print("=" * 110)
    print(detector)
    print("=" * 110)

    try:

        human = prepare(
            cfg["dir"]
            /
            cfg["human"],
            cfg["score_col"],
            cfg["sign"],
        )

        original = prepare(
            cfg["dir"]
            /
            cfg["original"],
            cfg["score_col"],
            cfg["sign"],
        )

        raw_o, canon_o, diff_o = verify(
            detector,
            "Original",
            "L1",
            human,
            original,
        )

        print(
            f"Original raw={raw_o:.6f} "
            f"canonical={canon_o:.6f} "
            f"diff={diff_o:.6f}"
        )

    except Exception as e:

        print(
            "FATAL:",
            e
        )

        audit.append({
            "Detector":
                detector,

            "Method":
                "Original",

            "Layer":
                "-",

            "Status":
                f"FAIL: {e}",
        })

        continue


    cache = {}


    for method in METHODS:

        for layer in LAYERS:

            cond = condition_name(
                method,
                layer,
            )

            path = (
                cfg["dir"]
                /
                cfg["pattern"].format(
                    condition=cond
                )
            )

            try:

                scores = prepare(
                    path,
                    cfg["score_col"],
                    cfg["sign"],
                )

                raw, canon, diff = verify(
                    detector,
                    method,
                    layer,
                    human,
                    scores,
                )

                cache[
                    (
                        method,
                        layer
                    )
                ] = scores

                seed_counter += 1

                r = bootstrap_delta(
                    human,
                    original,
                    scores,
                    N_BOOT,
                    SEED
                    +
                    seed_counter,
                )

                r.update({

                    "Detector":
                        detector,

                    "Method":
                        method,

                    "Layer":
                        layer,

                    "Canonical_AUROC":
                        canon,

                    "Raw_AUROC":
                        raw,

                    "Raw_vs_Canonical_diff":
                        diff,

                    "Raw_File":
                        str(
                            path
                        ),
                })

                results.append(
                    r
                )

                audit.append({

                    "Detector":
                        detector,

                    "Method":
                        method,

                    "Layer":
                        layer,

                    "Status":
                        "PASS",

                    "Raw_AUROC":
                        raw,

                    "Canonical_AUROC":
                        canon,

                    "Diff":
                        diff,
                })

                print(
                    f"{method:15s} {layer}: "
                    f"raw={raw:.4f} "
                    f"canonical={canon:.4f} "
                    f"delta={r['Delta_AUROC']:.4f} "
                    f"CI=[{r['Delta_CI_low']:.4f}, "
                    f"{r['Delta_CI_high']:.4f}]"
                )

            except Exception as e:

                print(
                    f"REJECT {method} {layer}: "
                    f"{e}"
                )

                audit.append({

                    "Detector":
                        detector,

                    "Method":
                        method,

                    "Layer":
                        layer,

                    "Status":
                        f"REJECTED: {e}",
                })


    # -------------------------------------------------------------
    # MATCHED BASELINE VS GRAMGUARD
    # -------------------------------------------------------------

    for layer in LAYERS:

        gg = cache.get(
            (
                "GramGuard",
                layer,
            )
        )

        if gg is None:
            continue

        for baseline in [
            "Qwen Humanize",
            "Pegasus",
        ]:

            b = cache.get(
                (
                    baseline,
                    layer,
                )
            )

            if b is None:
                continue

            seed_counter += 1

            r = bootstrap_difference(
                human,
                b,
                gg,
                N_BOOT,
                SEED
                +
                10000
                +
                seed_counter,
            )

            r.update({

                "Detector":
                    detector,

                "Layer":
                    layer,

                "Baseline":
                    baseline,
            })

            matched_results.append(
                r
            )


# =====================================================================
# SPECDETECT-LRR
#
# Each raw condition contains BOTH human_lrr and ai_lrr.
# =====================================================================

print()
print("=" * 110)
print("SpecDetect-LRR")
print("=" * 110)

SPEC_DIR = (
    BASE
    / "specdetect"
)


def spec_read(
    condition
):

    path = (
        SPEC_DIR
        /
        f"{condition}_lrr_raw_scores.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    required = {
        "index",
        "human_lrr",
        "ai_lrr",
    }

    if not required.issubset(
        df.columns
    ):

        raise RuntimeError(
            f"{path}: missing expected columns"
        )

    if len(df) != 900:

        raise RuntimeError(
            f"{path}: expected 900 rows, "
            f"found {len(df)}"
        )

    return df, path


def spec_auc(
    df,
    sign
):

    h = (
        sign
        *
        pd.to_numeric(
            df["human_lrr"],
            errors="coerce",
        )
    )

    a = (
        sign
        *
        pd.to_numeric(
            df["ai_lrr"],
            errors="coerce",
        )
    )

    valid = (
        h.notna()
        &
        a.notna()
    )

    h = h[
        valid
    ].to_numpy()

    a = a[
        valid
    ].to_numpy()

    y = np.r_[
        np.ones(len(a)),
        np.zeros(len(h)),
    ]

    s = np.r_[
        a,
        h,
    ]

    return roc_auc_score(
        y,
        s
    )


try:

    spec_orig, spec_orig_path = (
        spec_read(
            "original"
        )
    )

    canon_orig = canonical_auc(
        "SpecDetect-LRR",
        "Original",
        "L1",
    )

    direct = spec_auc(
        spec_orig,
        +1.0,
    )

    negated = spec_auc(
        spec_orig,
        -1.0,
    )

    spec_sign = (
        +1.0
        if
        abs(
            direct
            -
            canon_orig
        )
        <=
        abs(
            negated
            -
            canon_orig
        )
        else
        -1.0
    )

    raw_orig = spec_auc(
        spec_orig,
        spec_sign,
    )

    if abs(
        raw_orig
        -
        canon_orig
    ) > TOL:

        raise RuntimeError(
            f"Original raw={raw_orig:.6f} "
            f"canonical={canon_orig:.6f}"
        )

    print(
        "Orientation:",
        (
            "direct"
            if spec_sign > 0
            else "negated"
        )
    )

    print(
        f"Original raw={raw_orig:.6f} "
        f"canonical={canon_orig:.6f}"
    )


    spec_cache = {}


    for method in METHODS:

        for layer in LAYERS:

            cond = condition_name(
                method,
                layer,
            )

            try:

                df, path = spec_read(
                    cond
                )

                raw = spec_auc(
                    df,
                    spec_sign,
                )

                canon = canonical_auc(
                    "SpecDetect-LRR",
                    method,
                    layer,
                )

                diff = abs(
                    raw
                    -
                    canon
                )

                if diff > TOL:

                    raise RuntimeError(
                        f"raw={raw:.6f} does not "
                        f"reproduce canonical="
                        f"{canon:.6f}; "
                        f"diff={diff:.6f}"
                    )

                # Align on explicit index.
                joined = (
                    spec_orig[
                        [
                            "index",
                            "human_lrr",
                            "ai_lrr",
                        ]
                    ]
                    .rename(
                        columns={
                            "human_lrr":
                                "human_original",

                            "ai_lrr":
                                "ai_original",
                        }
                    )
                    .merge(
                        df[
                            [
                                "index",
                                "human_lrr",
                                "ai_lrr",
                            ]
                        ]
                        .rename(
                            columns={
                                "human_lrr":
                                    "human_condition",

                                "ai_lrr":
                                    "ai_condition",
                            }
                        ),
                        on="index",
                        how="inner",
                        validate="one_to_one",
                    )
                )

                if len(joined) != 900:

                    raise RuntimeError(
                        f"Only {len(joined)} "
                        "paired SpecDetect rows"
                    )

                ho = (
                    spec_sign
                    *
                    joined[
                        "human_original"
                    ].to_numpy(
                        dtype=float
                    )
                )

                hc = (
                    spec_sign
                    *
                    joined[
                        "human_condition"
                    ].to_numpy(
                        dtype=float
                    )
                )

                ao = (
                    spec_sign
                    *
                    joined[
                        "ai_original"
                    ].to_numpy(
                        dtype=float
                    )
                )

                ac = (
                    spec_sign
                    *
                    joined[
                        "ai_condition"
                    ].to_numpy(
                        dtype=float
                    )
                )

                seed_counter += 1

                rng = np.random.default_rng(
                    SEED
                    +
                    20000
                    +
                    seed_counter
                )

                deltas = np.empty(
                    N_BOOT
                )

                cond_boot = np.empty(
                    N_BOOT
                )

                n = len(joined)

                for i in range(
                    N_BOOT
                ):

                    ia = rng.integers(
                        0,
                        n,
                        size=n,
                    )

                    ih = rng.integers(
                        0,
                        n,
                        size=n,
                    )

                    y = np.r_[
                        np.ones(n),
                        np.zeros(n),
                    ]

                    auc_o = roc_auc_score(
                        y,
                        np.r_[
                            ao[ia],
                            ho[ih],
                        ]
                    )

                    auc_c = roc_auc_score(
                        y,
                        np.r_[
                            ac[ia],
                            hc[ih],
                        ]
                    )

                    cond_boot[i] = auc_c

                    deltas[i] = (
                        auc_o
                        -
                        auc_c
                    )

                results.append({

                    "Detector":
                        "SpecDetect-LRR",

                    "Method":
                        method,

                    "Layer":
                        layer,

                    "N_AI":
                        900,

                    "N_human":
                        900,

                    "Original_AUROC":
                        raw_orig,

                    "Condition_AUROC":
                        raw,

                    "Canonical_AUROC":
                        canon,

                    "Raw_AUROC":
                        raw,

                    "Raw_vs_Canonical_diff":
                        diff,

                    "Delta_AUROC":
                        raw_orig
                        -
                        raw,

                    "Condition_CI_low":
                        np.quantile(
                            cond_boot,
                            0.025,
                        ),

                    "Condition_CI_high":
                        np.quantile(
                            cond_boot,
                            0.975,
                        ),

                    "Delta_CI_low":
                        np.quantile(
                            deltas,
                            0.025,
                        ),

                    "Delta_CI_high":
                        np.quantile(
                            deltas,
                            0.975,
                        ),

                    "P_delta_gt_0":
                        np.mean(
                            deltas > 0
                        ),

                    "Raw_File":
                        str(
                            path
                        ),
                })

                spec_cache[
                    (
                        method,
                        layer
                    )
                ] = (
                    df,
                    raw
                )

                audit.append({

                    "Detector":
                        "SpecDetect-LRR",

                    "Method":
                        method,

                    "Layer":
                        layer,

                    "Status":
                        "PASS",

                    "Raw_AUROC":
                        raw,

                    "Canonical_AUROC":
                        canon,

                    "Diff":
                        diff,
                })

                print(
                    f"{method:15s} {layer}: "
                    f"raw={raw:.4f} "
                    f"canonical={canon:.4f} "
                    f"delta={raw_orig-raw:.4f}"
                )

            except Exception as e:

                print(
                    f"REJECT {method} {layer}: "
                    f"{e}"
                )

                audit.append({

                    "Detector":
                        "SpecDetect-LRR",

                    "Method":
                        method,

                    "Layer":
                        layer,

                    "Status":
                        f"REJECTED: {e}",
                })


except Exception as e:

    print(
        "SPECDETECT FATAL:",
        e
    )

    audit.append({

        "Detector":
            "SpecDetect-LRR",

        "Method":
            "Original",

        "Layer":
            "-",

        "Status":
            f"FAIL: {e}",
    })


# =====================================================================
# LASTDE / BINOCULARS STATUS
# =====================================================================

for detector in [
    "LastDE++",
    "Binoculars",
]:

    audit.append({

        "Detector":
            detector,

        "Method":
            "ALL",

        "Layer":
            "L1-L4",

        "Status":
            (
                "POINT ESTIMATES AVAILABLE; "
                "NO PER-SAMPLE RAW-SCORE CSV "
                "FOUND IN CURRENT INVENTORY; "
                "BOOTSTRAP CI NOT FABRICATED"
            ),
    })


# =====================================================================
# WRITE
# =====================================================================

res = pd.DataFrame(
    results
)

matched = pd.DataFrame(
    matched_results
)

audit_df = pd.DataFrame(
    audit
)


res.to_csv(
    OUT
    / "STRICT_detector_bootstrap.csv",
    index=False,
)


matched.to_csv(
    OUT
    / "STRICT_matched_baseline_bootstrap.csv",
    index=False,
)


audit_df.to_csv(
    OUT
    / "STRICT_bootstrap_audit.csv",
    index=False,
)


report = (
    OUT
    / "STRICT_FINAL_BOOTSTRAP_REPORT.txt"
)


with open(
    report,
    "w",
) as f:

    f.write(
        "GRAMGUARD-V3 STRICT FINAL "
        "DETECTOR BOOTSTRAP\n"
    )

    f.write(
        "=" * 110
        +
        "\n"
    )

    f.write(
        f"Bootstrap replicates: "
        f"{N_BOOT}\n"
    )

    f.write(
        f"Canonical verification tolerance: "
        f"{TOL}\n"
    )

    f.write(
        "Every accepted raw-score condition "
        "must reproduce canonical AUROC before "
        "its CI is reported.\n\n"
    )


    if not res.empty:

        cols = [

            "Detector",

            "Method",

            "Layer",

            "N_AI",

            "N_human",

            "Condition_AUROC",

            "Canonical_AUROC",

            "Delta_AUROC",

            "Delta_CI_low",

            "Delta_CI_high",

            "P_delta_gt_0",
        ]

        f.write(
            "ACCEPTED ORIGINAL -> CONDITION "
            "RESULTS\n"
        )

        f.write(
            "=" * 110
            +
            "\n"
        )

        f.write(
            res[
                [
                    c
                    for c in cols
                    if c in res.columns
                ]
            ]
            .sort_values(
                [
                    "Detector",
                    "Method",
                    "Layer",
                ]
            )
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        f.write(
            "\n\n"
        )


    if not matched.empty:

        f.write(
            "MATCHED BASELINE VS GRAMGUARD\n"
        )

        f.write(
            "=" * 110
            +
            "\n"
        )

        f.write(
            matched
            .sort_values(
                [
                    "Detector",
                    "Baseline",
                    "Layer",
                ]
            )
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        f.write(
            "\n\n"
        )


    f.write(
        "STRICT AUDIT\n"
    )

    f.write(
        "=" * 110
        +
        "\n"
    )

    f.write(
        audit_df.to_string(
            index=False
        )
    )

    f.write(
        "\n"
    )


print()
print("=" * 110)
print("STRICT BOOTSTRAP COMPLETE")
print("=" * 110)

print(
    "Output:",
    OUT
)

print(
    "Report:",
    report
)

print()
print(
    "PASS count:",
    (
        audit_df[
            "Status"
        ]
        .astype(str)
        .eq("PASS")
        .sum()
    )
)

print(
    "Rejected/failed count:",
    (
        audit_df[
            "Status"
        ]
        .astype(str)
        .str.contains(
            "REJECTED|FAIL",
            regex=True,
        )
        .sum()
    )
)

