#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GramGuard-V3: Final HRCG-Guided Human-Null Budget-Curve Attack
==============================================================

Final experiment modes: ``budget_curve`` (legacy BERT) and ``llm_budget_curve`` (final instruct-LLM test).

The attack uses fixed UMBC human-reference 2-gram/4-gram models. HRCG token
cross-order evidence localizes suspicious regions. A contextual masked-span
reconstructor generates multiple local alternatives. Candidates are filtered
only for basic quality/safety (semantic preservation, NLI, grammar, number and
negation preservation, and a cumulative edit-budget ceiling). The attacked
detectors are never queried during rewriting.

The candidate-selection objective is the original Idea-3 human-reference
alignment objective: minimize absolute distance of whole-document A24 from the
length-conditioned human-only null distribution. The human calibration table
is fixed and never recomputed from attacked texts.

A single nested trajectory is generated per document and snapshotted at hard
maximum changed-word budgets of 5%, 10%, 15%, and 20%. This makes the four
attack-strength conditions directly comparable while avoiding four independent
rewriting runs.

Legacy lexical/span modes remain in this file only for reproducibility. The
final experiment should use ``--mode budget_curve``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import kenlm
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.paths import (  # noqa: E402
    RAW_DATA_DIR,
    KENLM_2,
    KENLM_4,
    SEMANTIC_MODEL,
    OUTPUT_DIR,
)

from src.hrcg_core import clean_text, score_document, word_count  # noqa: E402


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# Editability prior. HRCG remains the attribution signal, while this prior
# prevents central topic nouns from dominating safer stylistic targets.
POS_EDITABILITY_WEIGHT = {
    "ADV": 1.40,
    "ADJ": 1.30,
    "VERB": 1.15,
    "NOUN": 0.45,
}

AUXILIARY_LEMMAS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
}

# Internal stopword set: deliberately conservative. We avoid downloading any
# external linguistic resource merely to run the candidate audit.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}

NEGATIONS = {"no", "not", "never", "neither", "nor", "without", "n't"}

WORD_CORE_RE = re.compile(r"^[A-Za-z]+(?:['’-][A-Za-z]+)*$")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")


# WordNet candidates are restricted to the target's coarse part of speech and
# to lemma names from a synset shared with the original word.
WORDNET_POS_BY_COARSE = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}

# These verbs are highly sensitive to complement structure. Even true lexical
# relatives may require different syntax (e.g., "let them go" vs.
# "allow them to go"). The precision-first attack therefore does not replace
# one member with another member of this set.
COMPLEMENT_SENSITIVE_VERBS = {
    "allow", "cause", "enable", "force", "have", "help", "let", "make",
    "permit", "require",
}

IRREGULAR_PAST = {
    "arise": "arose", "become": "became", "begin": "began", "break": "broke",
    "bring": "brought", "build": "built", "buy": "bought", "catch": "caught",
    "choose": "chose", "come": "came", "do": "did", "drive": "drove",
    "eat": "ate", "fall": "fell", "feel": "felt", "find": "found",
    "get": "got", "give": "gave", "go": "went", "grow": "grew",
    "have": "had", "hear": "heard", "hold": "held", "keep": "kept",
    "know": "knew", "lead": "led", "leave": "left", "lose": "lost",
    "make": "made", "meet": "met", "pay": "paid", "read": "read",
    "rise": "rose", "run": "ran", "say": "said", "see": "saw",
    "cut": "cut", "hit": "hit", "hurt": "hurt", "put": "put", "set": "set",
    "shut": "shut", "spread": "spread",
    "sell": "sold", "send": "sent", "sit": "sat", "speak": "spoke",
    "spend": "spent", "stand": "stood", "take": "took", "teach": "taught",
    "tell": "told", "think": "thought", "understand": "understood",
    "wear": "wore", "win": "won", "write": "wrote",
}

IRREGULAR_PARTICIPLE = {
    "arise": "arisen", "become": "become", "begin": "begun", "break": "broken",
    "bring": "brought", "build": "built", "buy": "bought", "catch": "caught",
    "choose": "chosen", "come": "come", "do": "done", "drive": "driven",
    "eat": "eaten", "fall": "fallen", "feel": "felt", "find": "found",
    "get": "gotten", "give": "given", "go": "gone", "grow": "grown",
    "have": "had", "hear": "heard", "hold": "held", "keep": "kept",
    "know": "known", "lead": "led", "leave": "left", "lose": "lost",
    "make": "made", "meet": "met", "pay": "paid", "read": "read",
    "rise": "risen", "run": "run", "say": "said", "see": "seen",
    "cut": "cut", "hit": "hit", "hurt": "hurt", "put": "put", "set": "set",
    "shut": "shut", "spread": "spread",
    "sell": "sold", "send": "sent", "sit": "sat", "speak": "spoken",
    "spend": "spent", "stand": "stood", "take": "taken", "teach": "taught",
    "tell": "told", "think": "thought", "understand": "understood",
    "wear": "worn", "win": "won", "write": "written",
}


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def normalize_label(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, str):
        value = x.strip().lower()
        if value in {"0", "human", "real", "original", "reference", "gold"}:
            return 0
        if value in {"1", "ai", "machine", "generated", "sampled", "model", "llm"}:
            return 1
        return None
    try:
        value = int(x)
        return value if value in {0, 1} else None
    except Exception:
        return None


def load_json_or_jsonl(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        rows: List[Any] = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def extract_texts(path: Path, source: str) -> List[str]:
    obj = load_json_or_jsonl(path)

    if isinstance(obj, dict):
        key = "sampled" if source == "ai" else "original"
        if isinstance(obj.get(key), list):
            return [clean_text(x) for x in obj[key] if isinstance(x, str) and clean_text(x)]

    wanted_label = 1 if source == "ai" else 0
    texts: List[str] = []
    if isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            label = None
            for key in ["label", "labels", "target", "class", "source"]:
                if key in row:
                    label = normalize_label(row.get(key))
                    if label is not None:
                        break
            if label != wanted_label:
                continue
            for key in [
                "text", "source_text", "completion", "response", "content",
                "sampled", "original", "human", "ai",
            ]:
                value = row.get(key)
                if isinstance(value, str) and clean_text(value):
                    texts.append(clean_text(value))
                    break
    return texts


# -----------------------------------------------------------------------------
# Token representation and exact HRCG attribution
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenParts:
    prefix: str
    core: str
    suffix: str


def split_token_parts(token: str) -> TokenParts:
    """Split punctuation from a whitespace token while retaining apostrophes."""
    match = re.match(
        r"^(?P<prefix>[^A-Za-z]*)(?P<core>[A-Za-z]+(?:['’-][A-Za-z]+)*)(?P<suffix>[^A-Za-z]*)$",
        token,
    )
    if not match:
        return TokenParts("", "", token)
    return TokenParts(match.group("prefix"), match.group("core"), match.group("suffix"))


def rebuild_token(parts: TokenParts, replacement_core: str) -> str:
    return f"{parts.prefix}{replacement_core}{parts.suffix}"


def replace_word_core(text: str, position: int, replacement_core: str) -> str:
    words = clean_text(text).split()
    if position < 0 or position >= len(words):
        raise IndexError(f"position={position}, words={len(words)}")
    parts = split_token_parts(words[position])
    if not parts.core:
        raise ValueError(f"Position {position} is not a lexical token: {words[position]!r}")
    words[position] = rebuild_token(parts, replacement_core)
    return " ".join(words)


def full_token_attribution(model2: kenlm.Model, model4: kenlm.Model, text: str) -> pd.DataFrame:
    text = clean_text(text)
    if not text:
        raise ValueError("Empty input text.")

    words = text.split()
    fs2 = list(model2.full_scores(text, bos=True, eos=True))
    fs4 = list(model4.full_scores(text, bos=True, eos=True))
    expected = len(words) + 1
    if len(fs2) != expected or len(fs4) != expected:
        raise RuntimeError(
            "KenLM token alignment mismatch: "
            f"words={len(words)}, 2gram_scores={len(fs2)}, 4gram_scores={len(fs4)}"
        )

    rows: List[Dict[str, Any]] = []
    for i, token in enumerate(words):
        logp2, used2, oov2 = fs2[i]
        logp4, used4, oov4 = fs4[i]
        ell2 = -float(logp2)
        ell4 = -float(logp4)
        parts = split_token_parts(token)
        rows.append(
            {
                "position": i,
                "token": token,
                "core": parts.core,
                "is_eos": False,
                "ell2": ell2,
                "ell4": ell4,
                "d24": ell2 - ell4,
                "abs_d24": abs(ell2 - ell4),
                "kenlm_order_used_2": int(used2),
                "kenlm_order_used_4": int(used4),
                "oov_2": bool(oov2),
                "oov_4": bool(oov4),
            }
        )

    logp2, used2, oov2 = fs2[-1]
    logp4, used4, oov4 = fs4[-1]
    ell2 = -float(logp2)
    ell4 = -float(logp4)
    rows.append(
        {
            "position": len(words),
            "token": "<EOS>",
            "core": "",
            "is_eos": True,
            "ell2": ell2,
            "ell4": ell4,
            "d24": ell2 - ell4,
            "abs_d24": abs(ell2 - ell4),
            "kenlm_order_used_2": int(used2),
            "kenlm_order_used_4": int(used4),
            "oov_2": bool(oov2),
            "oov_4": bool(oov4),
        }
    )
    return pd.DataFrame(rows)


def verify_attribution(doc: Dict[str, float], attribution: pd.DataFrame, text: str) -> None:
    length = max(1, word_count(text))
    reconstructed_gap = float(attribution["d24"].sum()) / length
    actual_gap = float(doc["H2"] - doc["H4"])
    reconstructed_a24 = reconstructed_gap / max(float(doc["H2"]), 1e-12)
    if not np.isclose(reconstructed_gap, actual_gap, rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            f"H2-H4 decomposition failed: reconstructed={reconstructed_gap}, actual={actual_gap}"
        )
    if not np.isclose(reconstructed_a24, float(doc["A24"]), rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            f"A24 decomposition failed: reconstructed={reconstructed_a24}, actual={doc['A24']}"
        )


def robust_center_scale(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return 0.0, 1.0
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    # 1.4826 makes MAD comparable to standard deviation under normality.
    scale = max(1.4826 * mad, 1e-6)
    return center, scale


# -----------------------------------------------------------------------------
# Optional part-of-speech tagging
# -----------------------------------------------------------------------------


class OptionalPosTagger:
    def __init__(self) -> None:
        self.available = False
        self.reason = ""
        self._nltk = None
        try:
            import nltk  # type: ignore

            # A one-token call is enough to test both package and tagger data.
            nltk.pos_tag(["test"])
            self._nltk = nltk
            self.available = True
        except Exception as exc:  # package absent or tagger resource missing
            self.reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__

    @staticmethod
    def coarse(tag: str) -> str:
        if tag.startswith("NN"):
            return "NOUN"
        if tag.startswith("VB"):
            return "VERB"
        if tag.startswith("JJ"):
            return "ADJ"
        if tag.startswith("RB"):
            return "ADV"
        return "OTHER"

    def tag_words(self, words: Sequence[str]) -> List[str]:
        if not self.available or self._nltk is None:
            raise RuntimeError(
                "Mandatory NLTK POS tagging is unavailable. Install the local "
                "tagger resource before running GramGuard-V3 synonym-constrained lexical method."
            )
        cleaned = [split_token_parts(word).core or word for word in words]
        try:
            tags = [tag for _, tag in self._nltk.pos_tag(cleaned)]
        except Exception as exc:
            raise RuntimeError(f"NLTK POS tagging failed: {exc}") from exc
        if len(tags) != len(words) or any(not tag for tag in tags):
            raise RuntimeError("NLTK POS tagging returned incomplete output.")
        return tags


# -----------------------------------------------------------------------------
# Target-word selection
# -----------------------------------------------------------------------------


def is_probable_named_entity(words: Sequence[str], position: int, core: str) -> bool:
    if core.isupper() and len(core) > 1:
        return True
    # Capitalized words away from a sentence start are conservatively protected.
    if core[:1].isupper():
        if position == 0:
            return False
        previous = words[position - 1]
        if re.search(r"[.!?][\"')\]]*$", previous):
            return False
        return True
    return False


def lexical_target_ok(
    words: Sequence[str],
    position: int,
    core: str,
    pos_tag: str,
    pos_tagger: OptionalPosTagger,
    min_word_length: int,
    token_frequency: Dict[str, int],
) -> Tuple[bool, str]:
    if not core or not WORD_CORE_RE.fullmatch(core):
        return False, "non_lexical"

    lower = core.lower()
    if len(lower) < min_word_length:
        return False, "too_short"
    if lower in STOPWORDS:
        return False, "stopword"
    if lower in AUXILIARY_LEMMAS:
        return False, "auxiliary"
    if any(char.isdigit() for char in core):
        return False, "contains_number"
    if lower.endswith("'s") or lower.endswith("’s"):
        return False, "possessive"
    if is_probable_named_entity(words, position, core):
        return False, "probable_named_entity"

    coarse = pos_tagger.coarse(pos_tag)
    if coarse not in CONTENT_POS:
        return False, f"non_content_pos:{pos_tag}"

    # Repeated nouns are often topic-bearing terms (e.g., orange, blood,
    # diabetes). Protect them because changing them can preserve sentence-level
    # embeddings while altering the document's subject or factual content.
    if coarse == "NOUN" and token_frequency.get(lower, 0) >= 2:
        return False, "repeated_topic_noun"

    return True, ""


def select_hrcg_targets(
    attribution: pd.DataFrame,
    text: str,
    pos_tagger: OptionalPosTagger,
    target_count: int,
    min_word_length: int,
    excluded_positions: Optional[set[int]] = None,
) -> Tuple[pd.DataFrame, float, float]:
    words = clean_text(text).split()
    pos_tags = pos_tagger.tag_words(words)
    textual = attribution.loc[~attribution["is_eos"]].copy()
    center, scale = robust_center_scale(textual["d24"].to_numpy(dtype=float))
    textual["hrcg_center"] = center
    textual["hrcg_scale"] = scale
    textual["hrcg_anomaly"] = np.abs(textual["d24"] - center) / scale

    token_frequency: Dict[str, int] = {}
    for word in words:
        core = split_token_parts(word).core.lower()
        if core:
            token_frequency[core] = token_frequency.get(core, 0) + 1

    excluded_positions = excluded_positions or set()
    editable: List[bool] = []
    reasons: List[str] = []
    coarse_tags: List[str] = []
    fine_tags: List[str] = []
    pos_weights: List[float] = []
    frequencies: List[int] = []

    for row in textual.itertuples(index=False):
        position = int(row.position)
        pos_tag = pos_tags[position]
        coarse = pos_tagger.coarse(pos_tag)
        core = str(row.core)
        lower = core.lower()

        if position in excluded_positions:
            ok, reason = False, "already_edited"
        else:
            ok, reason = lexical_target_ok(
                words,
                position,
                core,
                pos_tag,
                pos_tagger,
                min_word_length,
                token_frequency,
            )

        editable.append(ok)
        reasons.append(reason)
        fine_tags.append(pos_tag)
        coarse_tags.append(coarse)
        pos_weights.append(float(POS_EDITABILITY_WEIGHT.get(coarse, 0.0)))
        frequencies.append(int(token_frequency.get(lower, 0)))

    textual["pos_tag"] = fine_tags
    textual["coarse_pos"] = coarse_tags
    textual["token_frequency"] = frequencies
    textual["pos_editability_weight"] = pos_weights
    textual["target_score"] = textual["hrcg_anomaly"] * textual["pos_editability_weight"]
    textual["editable"] = editable
    textual["target_exclusion_reason"] = reasons

    selected = (
        textual.loc[textual["editable"]]
        .sort_values(
            ["target_score", "hrcg_anomaly", "abs_d24"],
            ascending=[False, False, False],
        )
        .head(int(target_count))
        .reset_index(drop=True)
    )
    selected.insert(0, "target_rank", range(1, len(selected) + 1))
    return selected, center, scale


# -----------------------------------------------------------------------------
# Sentence/local context boundaries
# -----------------------------------------------------------------------------


def sentence_bounds(words: Sequence[str], position: int) -> Tuple[int, int]:
    start = 0
    for i in range(position - 1, -1, -1):
        if re.search(r"[.!?][\"')\]]*$", words[i]):
            start = i + 1
            break

    end = len(words) - 1
    for i in range(position, len(words)):
        if re.search(r"[.!?][\"')\]]*$", words[i]):
            end = i
            break
    return start, end


def window_bounds(length: int, position: int, radius: int) -> Tuple[int, int]:
    return max(0, position - radius), min(length - 1, position + radius)


def slice_words(words: Sequence[str], start: int, end: int) -> str:
    return " ".join(words[start : end + 1])


def kenlm_cross_entropy(model: kenlm.Model, text: str) -> float:
    text = clean_text(text)
    if not text:
        return float("inf")
    return -float(model.score(text, bos=True, eos=True)) / max(1, word_count(text))


# -----------------------------------------------------------------------------
# Local masked-language-model discovery and loading
# -----------------------------------------------------------------------------


def model_has_weights(snapshot: Path) -> bool:
    patterns = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model-*.bin",
        "model-*.safetensors",
    ]
    return any(any(snapshot.glob(pattern)) for pattern in patterns)


def discover_local_masked_lms(cache_root: Optional[Path] = None) -> pd.DataFrame:
    cache_root = cache_root or (Path.home() / ".cache" / "huggingface" / "hub")
    rows: List[Dict[str, Any]] = []
    if not cache_root.exists():
        return pd.DataFrame(rows)

    preferred_types = {
        "bert", "roberta", "deberta", "deberta-v2", "distilbert", "albert",
        "electra", "xlm-roberta", "camembert",
    }

    for config_path in cache_root.glob("models--*/snapshots/*/config.json"):
        snapshot = config_path.parent
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_type = str(config.get("model_type", ""))
        architectures = [str(x) for x in config.get("architectures", [])]
        masked_arch = any("MaskedLM" in architecture for architecture in architectures)
        type_candidate = model_type in preferred_types
        if not (masked_arch or type_candidate):
            continue
        has_tokenizer = any(
            (snapshot / name).exists()
            for name in [
                "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                "vocab.json", "sentencepiece.bpe.model", "spiece.model",
            ]
        )
        rows.append(
            {
                "model_dir": snapshot.parent.parent.name,
                "snapshot": str(snapshot),
                "model_type": model_type,
                "architectures": "|".join(architectures),
                "has_weights": model_has_weights(snapshot),
                "has_tokenizer": has_tokenizer,
                "is_explicit_masked_lm": masked_arch,
            }
        )

    if not rows:
        return pd.DataFrame(rows)
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["is_explicit_masked_lm", "has_weights", "has_tokenizer", "model_type"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def resolve_mlm_path(requested: str) -> Path:
    if requested and requested.lower() != "auto":
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Masked-LM path does not exist: {path}")
        return path

    env_value = os.environ.get("GRAMGUARD_MLM_MODEL", "").strip()
    if env_value:
        path = Path(env_value).expanduser().resolve()
        if path.exists():
            return path

    discovered = discover_local_masked_lms()
    if discovered.empty:
        raise FileNotFoundError(
            "No local masked language model was discovered. Run:\n"
            "  python scripts/gramguard_v3.py --mode discover\n"
            "Then provide --mlm_model /exact/local/snapshot/path."
        )

    usable = discovered.loc[discovered["has_weights"] & discovered["has_tokenizer"]]
    if usable.empty:
        raise FileNotFoundError(
            "Masked-LM-like cache directories were found, but none had both weights and tokenizer files. "
            "Run --mode discover and inspect the table."
        )
    return Path(str(usable.iloc[0]["snapshot"])).resolve()


def choose_torch_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_masked_lm(model_path: Path, device: torch.device):
    print(f"[LOAD] Masked LM tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.mask_token_id is None or tokenizer.mask_token is None:
        raise RuntimeError(f"Tokenizer has no mask token: {model_path}")

    print(f"[LOAD] Masked LM model    : {model_path}")
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    print(f"[LOAD] Masked LM device   : {device}")
    return tokenizer, model


def latest_local_snapshot(repo_cache_name: str) -> Optional[Path]:
    snapshots_root = Path.home() / ".cache" / "huggingface" / "hub" / repo_cache_name / "snapshots"
    if not snapshots_root.exists():
        return None
    snapshots = sorted(
        [path for path in snapshots_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0].resolve() if snapshots else None


def resolve_validation_model_path(
    requested: str,
    *,
    env_name: str,
    repo_cache_name: str,
    display_name: str,
) -> Path:
    if requested and requested.lower() != "auto":
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{display_name} path does not exist: {path}")
        return path

    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        path = Path(env_value).expanduser().resolve()
        if path.exists():
            return path

    discovered = latest_local_snapshot(repo_cache_name)
    if discovered is None:
        raise FileNotFoundError(
            f"No local {display_name} snapshot was found under the Hugging Face cache. "
            f"Provide --{env_name.lower()} /exact/snapshot/path or download the required model."
        )
    return discovered


def rough_lemma(word: str) -> str:
    core = split_token_parts(word).core.lower()
    if len(core) > 4 and core.endswith("ies"):
        return core[:-3] + "y"
    if len(core) > 4 and core.endswith("ing"):
        stem = core[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(core) > 3 and core.endswith("ed"):
        stem = core[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(core) > 4 and core.endswith(("ches", "shes", "xes", "zes", "ses", "oes")):
        return core[:-2]
    if len(core) > 3 and core.endswith("s") and not core.endswith("ss"):
        return core[:-1]
    return core


def adjacent_lemma_duplication_ok(words_after: Sequence[str], position: int) -> bool:
    current = rough_lemma(words_after[position])
    if not current:
        return True
    for neighbor in (position - 1, position + 1):
        if 0 <= neighbor < len(words_after) and rough_lemma(words_after[neighbor]) == current:
            return False
    return True


class SemanticEquivalenceGate:
    """Conservative bidirectional NLI + CoLA validator for final candidates."""

    def __init__(
        self,
        *,
        nli_path: Path,
        cola_path: Path,
        device: torch.device,
        min_entailment: float,
        max_contradiction: float,
        min_cola_acceptability: float,
        max_cola_drop: float,
    ) -> None:
        self.device = device
        self.min_entailment = float(min_entailment)
        self.max_contradiction = float(max_contradiction)
        self.min_cola_acceptability = float(min_cola_acceptability)
        self.max_cola_drop = float(max_cola_drop)

        print(f"[LOAD] NLI tokenizer      : {nli_path}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            str(nli_path), local_files_only=True
        )
        print(f"[LOAD] NLI model          : {nli_path}")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            str(nli_path), local_files_only=True
        )
        self.nli_model.to(device)
        self.nli_model.eval()

        print(f"[LOAD] CoLA tokenizer     : {cola_path}")
        self.cola_tokenizer = AutoTokenizer.from_pretrained(
            str(cola_path), local_files_only=True
        )
        print(f"[LOAD] CoLA model         : {cola_path}")
        self.cola_model = AutoModelForSequenceClassification.from_pretrained(
            str(cola_path), local_files_only=True
        )
        self.cola_model.to(device)
        self.cola_model.eval()
        print(f"[LOAD] Validation device  : {device}")

        labels = {
            int(index): str(label).lower()
            for index, label in self.nli_model.config.id2label.items()
        }
        self.entailment_id = next(
            (index for index, label in labels.items() if "entail" in label), 0
        )
        self.contradiction_id = next(
            (index for index, label in labels.items() if "contrad" in label), 2
        )
        if self.nli_model.config.num_labels < 3:
            raise RuntimeError("The selected NLI model does not expose three NLI labels.")
        if self.cola_model.config.num_labels != 2:
            raise RuntimeError("The selected CoLA model must expose exactly two labels.")

    def _nli_probabilities(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> np.ndarray:
        encoded = self.nli_tokenizer(
            list(premises),
            list(hypotheses),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.nli_model(**encoded).logits, dim=-1)
        return probabilities.detach().cpu().numpy()

    def _cola_probabilities(self, sentences: Sequence[str]) -> np.ndarray:
        encoded = self.cola_tokenizer(
            list(sentences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.cola_model(**encoded).logits, dim=-1)
        # GLUE CoLA convention: class 1 = linguistically acceptable.
        return probabilities[:, 1].detach().cpu().numpy()

    def score(
        self,
        originals: Sequence[str],
        candidates: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if len(originals) != len(candidates):
            raise ValueError("NLI/CoLA original and candidate batches must match.")
        if not originals:
            return []

        forward = self._nli_probabilities(originals, candidates)
        backward = self._nli_probabilities(candidates, originals)
        original_cola = self._cola_probabilities(originals)
        candidate_cola = self._cola_probabilities(candidates)

        results: List[Dict[str, Any]] = []
        for index in range(len(originals)):
            forward_entailment = float(forward[index, self.entailment_id])
            backward_entailment = float(backward[index, self.entailment_id])
            forward_contradiction = float(forward[index, self.contradiction_id])
            backward_contradiction = float(backward[index, self.contradiction_id])
            min_entailment = min(forward_entailment, backward_entailment)
            max_contradiction = max(forward_contradiction, backward_contradiction)
            original_acceptability = float(original_cola[index])
            candidate_acceptability = float(candidate_cola[index])
            cola_drop = original_acceptability - candidate_acceptability

            nli_ok = (
                min_entailment >= self.min_entailment
                and max_contradiction <= self.max_contradiction
            )
            cola_ok = (
                candidate_acceptability >= self.min_cola_acceptability
                and cola_drop <= self.max_cola_drop
            )
            results.append(
                {
                    "nli_forward_entailment": forward_entailment,
                    "nli_backward_entailment": backward_entailment,
                    "nli_min_bidirectional_entailment": min_entailment,
                    "nli_forward_contradiction": forward_contradiction,
                    "nli_backward_contradiction": backward_contradiction,
                    "nli_max_bidirectional_contradiction": max_contradiction,
                    "nli_ok": bool(nli_ok),
                    "cola_original_acceptability": original_acceptability,
                    "cola_candidate_acceptability": candidate_acceptability,
                    "cola_acceptability_drop": cola_drop,
                    "cola_ok": bool(cola_ok),
                    "semantic_equivalence_ok": bool(nli_ok and cola_ok),
                }
            )
        return results


# -----------------------------------------------------------------------------
# Synonym-constrained candidate generation
# -----------------------------------------------------------------------------


def preserve_case(candidate: str, original: str) -> str:
    if original.isupper():
        return candidate.upper()
    if original[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate.lower()


def normalize_decoded_token(tokenizer, token_id: int) -> str:
    token = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    token = token.replace("Ġ", "").replace("▁", "").strip()
    return token


def build_masked_context(
    text: str,
    position: int,
    mask_token: str,
    max_context_words: int,
) -> Tuple[str, int, int]:
    words = clean_text(text).split()
    sentence_start, sentence_end = sentence_bounds(words, position)

    if sentence_end - sentence_start + 1 > max_context_words:
        half = max(4, max_context_words // 2)
        sentence_start = max(sentence_start, position - half)
        sentence_end = min(sentence_end, position + half)

    parts = split_token_parts(words[position])
    masked_words = list(words[sentence_start : sentence_end + 1])
    relative = position - sentence_start
    masked_words[relative] = rebuild_token(parts, mask_token)
    return " ".join(masked_words), sentence_start, sentence_end


def _is_consonant(char: str) -> bool:
    return len(char) == 1 and char.isalpha() and char.lower() not in "aeiou"


def _ends_cvc(word: str) -> bool:
    lower = word.lower()
    return (
        len(lower) >= 3
        and _is_consonant(lower[-3])
        and lower[-2] in "aeiou"
        and _is_consonant(lower[-1])
        and lower[-1] not in "wxy"
    )


def _third_person_singular(lemma: str) -> str:
    lower = lemma.lower()
    if lower.endswith("y") and len(lower) > 1 and _is_consonant(lower[-2]):
        return lower[:-1] + "ies"
    if lower.endswith(("s", "x", "z", "ch", "sh", "o")):
        return lower + "es"
    return lower + "s"


def _pluralize(lemma: str) -> str:
    lower = lemma.lower()
    irregular = {
        "child": "children", "foot": "feet", "goose": "geese", "man": "men",
        "mouse": "mice", "person": "people", "tooth": "teeth", "woman": "women",
    }
    if lower in irregular:
        return irregular[lower]
    if lower.endswith("y") and len(lower) > 1 and _is_consonant(lower[-2]):
        return lower[:-1] + "ies"
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return lower + "es"
    if lower.endswith("fe"):
        return lower[:-2] + "ves"
    if lower.endswith("f"):
        return lower[:-1] + "ves"
    return lower + "s"


def _regular_past(lemma: str) -> str:
    lower = lemma.lower()
    if lower.endswith("e"):
        return lower + "d"
    if lower.endswith("y") and len(lower) > 1 and _is_consonant(lower[-2]):
        return lower[:-1] + "ied"
    if _ends_cvc(lower):
        return lower + lower[-1] + "ed"
    return lower + "ed"


def _present_participle(lemma: str) -> str:
    lower = lemma.lower()
    if lower.endswith("ie"):
        return lower[:-2] + "ying"
    if lower.endswith("e") and not lower.endswith(("ee", "ye", "oe")):
        return lower[:-1] + "ing"
    if _ends_cvc(lower):
        return lower + lower[-1] + "ing"
    return lower + "ing"


def inflect_wordnet_lemma(lemma: str, penn_tag: str) -> Optional[str]:
    """Return a conservative one-word surface form matching the target tag."""
    base = lemma.lower().strip()
    if not base or not WORD_CORE_RE.fullmatch(base):
        return None

    if penn_tag in {"NN", "NNP", "JJ", "RB", "VB", "VBP"}:
        return base
    if penn_tag in {"NNS", "NNPS"}:
        return _pluralize(base)
    if penn_tag == "VBZ":
        return _third_person_singular(base)
    if penn_tag == "VBG":
        return _present_participle(base)
    if penn_tag == "VBD":
        return IRREGULAR_PAST.get(base, _regular_past(base))
    if penn_tag == "VBN":
        return IRREGULAR_PARTICIPLE.get(base, _regular_past(base))

    # Comparative and superlative morphology is not reliably transferable
    # between arbitrary WordNet lemmas in a single-word attack.
    if penn_tag in {"JJR", "JJS", "RBR", "RBS"}:
        return None
    return None


class WordNetSynonymProvider:
    """Precision-first same-synset lexical candidate resource."""

    def __init__(self) -> None:
        try:
            from nltk.corpus import wordnet as wn  # type: ignore

            wn.ensure_loaded()
        except LookupError as exc:
            raise RuntimeError(
                "NLTK WordNet is required for synonym-constrained GramGuard-V3. "
                "Install it once in the active environment with:\n"
                "  python - <<'PY'\n"
                "  import nltk\n"
                "  nltk.download('wordnet')\n"
                "  nltk.download('omw-1.4')\n"
                "  PY"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Could not load NLTK WordNet: {exc}") from exc
        self.wn = wn

    def candidates(
        self,
        original_core: str,
        penn_tag: str,
        coarse_pos: str,
    ) -> List[Dict[str, Any]]:
        wn_pos = WORDNET_POS_BY_COARSE.get(coarse_pos)
        if wn_pos is None:
            return []

        original_lower = original_core.lower()
        synsets = self.wn.synsets(original_lower, pos=wn_pos)
        if not synsets and wn_pos == "a":
            # WordNet distinguishes adjective satellites internally; querying
            # with adjective POS normally covers them, but this fallback is
            # harmless across NLTK versions.
            synsets = self.wn.synsets(original_lower, pos=self.wn.ADJ)
        if not synsets:
            return []

        original_lemma = self.wn.morphy(original_lower, wn_pos) or rough_lemma(original_lower)
        by_surface: Dict[str, Dict[str, Any]] = {}

        for synset in synsets:
            for lemma_obj in synset.lemmas():
                candidate_lemma = lemma_obj.name().replace("_", "-").lower()
                # Multiword expressions are outside the locked one-word method.
                if "_" in lemma_obj.name() or " " in candidate_lemma:
                    continue
                if candidate_lemma == original_lemma or candidate_lemma == original_lower:
                    continue
                surface = inflect_wordnet_lemma(candidate_lemma, penn_tag)
                if surface is None:
                    continue
                if rough_lemma(surface) == rough_lemma(original_lower):
                    continue
                if surface in STOPWORDS or not WORD_CORE_RE.fullmatch(surface):
                    continue

                record = by_surface.setdefault(
                    surface,
                    {
                        "candidate_core": preserve_case(surface, original_core),
                        "candidate_lemma": candidate_lemma,
                        "original_lemma": original_lemma,
                        "wordnet_pos": wn_pos,
                        "wordnet_shared_synsets": [],
                        "candidate_source": "wordnet_same_synset",
                        "wordnet_relation_ok": True,
                    },
                )
                record["wordnet_shared_synsets"].append(synset.name())

        results: List[Dict[str, Any]] = []
        for record in by_surface.values():
            shared = sorted(set(record["wordnet_shared_synsets"]))
            record["wordnet_shared_synsets"] = "|".join(shared)
            record["wordnet_shared_synset_count"] = len(shared)
            results.append(record)

        return sorted(
            results,
            key=lambda item: (
                -int(item["wordnet_shared_synset_count"]),
                str(item["candidate_core"]).lower(),
            ),
        )


def lexical_frame_compatibility_ok(
    original_lemma: str,
    candidate_lemma: str,
    coarse_pos: str,
) -> bool:
    if coarse_pos != "VERB":
        return True
    original = rough_lemma(original_lemma)
    candidate = rough_lemma(candidate_lemma)
    if original == candidate:
        return True
    if original in COMPLEMENT_SENSITIVE_VERBS or candidate in COMPLEMENT_SENSITIVE_VERBS:
        return False
    return True


def generate_synonym_candidates(
    tokenizer,
    model,
    device: torch.device,
    synonym_provider: WordNetSynonymProvider,
    text: str,
    position: int,
    penn_tag: str,
    coarse_pos: str,
    top_k: int,
    max_context_words: int,
) -> List[Dict[str, Any]]:
    words = clean_text(text).split()
    original_core = split_token_parts(words[position]).core
    resource_candidates = synonym_provider.candidates(
        original_core=original_core,
        penn_tag=penn_tag,
        coarse_pos=coarse_pos,
    )
    if not resource_candidates:
        return []

    masked_text, _, _ = build_masked_context(
        text,
        position,
        tokenizer.mask_token,
        max_context_words,
    )
    encoded = tokenizer(
        masked_text,
        return_tensors="pt",
        truncation=True,
        max_length=min(int(getattr(tokenizer, "model_max_length", 512)), 512),
    )
    mask_positions = (
        encoded["input_ids"][0] == tokenizer.mask_token_id
    ).nonzero(as_tuple=False)
    if mask_positions.numel() != 1:
        raise RuntimeError(
            f"Expected exactly one mask token, found {mask_positions.numel()} in: {masked_text!r}"
        )

    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits[0, int(mask_positions.item()), :]
        log_probs = torch.log_softmax(logits, dim=-1)

    scored: List[Dict[str, Any]] = []
    for candidate in resource_candidates:
        candidate_core = str(candidate["candidate_core"])
        token_ids = tokenizer.encode(candidate_core, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = int(token_ids[0])
        if token_id in set(getattr(tokenizer, "all_special_ids", [])):
            continue
        if getattr(tokenizer, "unk_token_id", None) == token_id:
            continue

        decoded = normalize_decoded_token(tokenizer, token_id)
        if decoded.lower() != candidate_core.lower():
            continue

        log_probability = float(log_probs[token_id].detach().cpu())
        global_rank = int((log_probs > log_probs[token_id]).sum().detach().cpu()) + 1
        record = dict(candidate)
        record.update(
            {
                "mlm_log_probability": log_probability,
                "mlm_probability": float(math.exp(log_probability)),
                "mlm_global_rank": global_rank,
                "wordnet_pool_size": len(resource_candidates),
            }
        )
        scored.append(record)

    scored.sort(
        key=lambda item: (
            -float(item["mlm_log_probability"]),
            -int(item["wordnet_shared_synset_count"]),
            str(item["candidate_core"]).lower(),
        )
    )
    for rank, candidate in enumerate(scored, start=1):
        candidate["mlm_rank"] = rank
    return scored[: int(top_k)]


# -----------------------------------------------------------------------------
# Linguistic and semantic validation
# -----------------------------------------------------------------------------


def number_signature(text: str) -> List[str]:
    return NUMBER_RE.findall(clean_text(text))


def negation_signature(text: str) -> List[str]:
    lowered = clean_text(text).lower()
    tokens = re.findall(r"[a-z]+(?:n't)?", lowered)
    return [token for token in tokens if token in NEGATIONS or token.endswith("n't")]


def suffix_class(word: str) -> str:
    lower = word.lower()
    for suffix in ["ingly", "edly", "ing", "ied", "ed", "ly", "iest", "ier", "est", "er"]:
        if len(lower) > len(suffix) + 2 and lower.endswith(suffix):
            return suffix
    if len(lower) > 3 and lower.endswith("ies"):
        return "plural_ies"
    if len(lower) > 3 and lower.endswith("es"):
        return "plural_es"
    if len(lower) > 3 and lower.endswith("s") and not lower.endswith("ss"):
        return "plural_s"
    return "base"


def heuristic_morphology_ok(original: str, candidate: str) -> bool:
    original_class = suffix_class(original)
    candidate_class = suffix_class(candidate)
    if original_class == "base":
        return candidate_class == "base"
    return original_class == candidate_class


def pos_and_morphology_ok(
    pos_tagger: OptionalPosTagger,
    words_before: Sequence[str],
    words_after: Sequence[str],
    position: int,
    original_core: str,
    candidate_core: str,
) -> Tuple[bool, bool, str, str]:
    if not pos_tagger.available:
        raise RuntimeError("Mandatory POS tagging became unavailable during candidate evaluation.")

    before_tags = pos_tagger.tag_words(words_before)
    after_tags = pos_tagger.tag_words(words_after)
    before_tag = before_tags[position] if position < len(before_tags) else ""
    after_tag = after_tags[position] if position < len(after_tags) else ""
    coarse_ok = (
        bool(before_tag)
        and bool(after_tag)
        and pos_tagger.coarse(before_tag) == pos_tagger.coarse(after_tag)
        and pos_tagger.coarse(before_tag) in CONTENT_POS
    )

    # Fine-grained morphology matters most for verbs, number, and comparatives.
    fine_sensitive = {
        "VBD", "VBG", "VBN", "VBP", "VBZ", "NNS", "NNPS", "JJR", "JJS", "RBR", "RBS"
    }
    if before_tag in fine_sensitive:
        morphology_ok = after_tag == before_tag
    else:
        morphology_ok = coarse_ok and heuristic_morphology_ok(original_core, candidate_core)
    return coarse_ok, morphology_ok, before_tag, after_tag


def adjacent_duplication_ok(words_after: Sequence[str], position: int) -> bool:
    current = split_token_parts(words_after[position]).core.lower()
    if position > 0 and split_token_parts(words_after[position - 1]).core.lower() == current:
        return False
    if position + 1 < len(words_after) and split_token_parts(words_after[position + 1]).core.lower() == current:
        return False
    return True


def semantic_scores(
    semantic_model: SentenceTransformer,
    originals: Sequence[str],
    candidates: Sequence[str],
) -> List[float]:
    if len(originals) != len(candidates):
        raise ValueError("original/candidate semantic batches must have equal length")
    if not originals:
        return []
    texts: List[str] = []
    for original, candidate in zip(originals, candidates):
        texts.extend([original, candidate])
    embeddings = semantic_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [
        float(np.dot(embeddings[2 * i], embeddings[2 * i + 1]))
        for i in range(len(originals))
    ]


# -----------------------------------------------------------------------------
# Candidate evaluation
# -----------------------------------------------------------------------------


def local_anomaly_sum(
    attribution: pd.DataFrame,
    start: int,
    end: int,
    center: float,
    scale: float,
) -> float:
    values = attribution.loc[
        (~attribution["is_eos"])
        & (attribution["position"] >= start)
        & (attribution["position"] <= end),
        "d24",
    ].to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    return float(np.sum(np.abs(values - center) / max(scale, 1e-6)))


def rejection_reason(row: Dict[str, Any]) -> str:
    reasons: List[str] = []
    checks = [
        ("wordnet_relation_ok", "not_a_same_synset_wordnet_candidate"),
        ("frame_compatibility_ok", "lexical_frame_incompatible"),
        ("mlm_context_ok", "low_probability_or_ranked_mlm_candidate"),
        ("local_context_semantic_ok", "local_context_semantic_below_threshold"),
        ("sentence_semantic_ok", "sentence_semantic_below_threshold"),
        ("pos_ok", "part_of_speech_changed"),
        ("morphology_ok", "morphology_changed"),
        ("numbers_preserved", "numbers_changed"),
        ("negation_preserved", "negation_changed"),
        ("adjacent_duplication_ok", "adjacent_duplication"),
        ("adjacent_lemma_duplication_ok", "adjacent_lemma_duplication"),
        ("local_h4_improved", "no_local_H4_improvement"),
        ("document_h4_ok", "document_H4_degraded"),
    ]
    for key, message in checks:
        if not bool(row.get(key, False)):
            reasons.append(message)

    if bool(row.get("semantic_validation_applied", False)):
        if not bool(row.get("nli_ok", False)):
            reasons.append("bidirectional_NLI_failed")
        if not bool(row.get("cola_ok", False)):
            reasons.append("grammatical_acceptability_failed")
    return ";".join(reasons)


def evaluate_target_candidates(
    *,
    text: str,
    target_row: pd.Series,
    raw_candidates: List[Dict[str, float]],
    model2: kenlm.Model,
    model4: kenlm.Model,
    semantic_model: SentenceTransformer,
    pos_tagger: OptionalPosTagger,
    current_doc: Dict[str, float],
    current_attribution: pd.DataFrame,
    center: float,
    scale: float,
    local_radius: int,
    local_context_semantic_threshold: float,
    sentence_semantic_threshold: float,
    max_mlm_rank: int,
    max_mlm_global_rank: int,
    min_mlm_probability: float,
    min_local_h4_gain: float,
    max_document_h4_increase: float,
    semantic_gate: SemanticEquivalenceGate,
) -> List[Dict[str, Any]]:
    if not raw_candidates:
        return []

    words_before = clean_text(text).split()
    position = int(target_row["position"])
    original_token = words_before[position]
    original_parts = split_token_parts(original_token)
    sentence_start, sentence_end = sentence_bounds(words_before, position)
    local_start, local_end = window_bounds(len(words_before), position, local_radius)
    original_sentence = slice_words(words_before, sentence_start, sentence_end)
    original_local = slice_words(words_before, local_start, local_end)
    original_local_h4 = kenlm_cross_entropy(model4, original_local)

    # Counterfactual HRCG anomaly is retained for analysis only. It never
    # controls acceptance or ranking in the locked synonym-constrained method.
    anomaly_before = local_anomaly_sum(
        current_attribution, local_start, local_end, center, scale
    )

    candidate_texts: List[str] = []
    candidate_sentences: List[str] = []
    candidate_locals: List[str] = []
    candidate_words: List[List[str]] = []
    for candidate in raw_candidates:
        candidate_text = replace_word_core(text, position, str(candidate["candidate_core"]))
        words_after = candidate_text.split()
        candidate_texts.append(candidate_text)
        candidate_sentences.append(slice_words(words_after, sentence_start, sentence_end))
        candidate_locals.append(slice_words(words_after, local_start, local_end))
        candidate_words.append(words_after)

    local_context_similarities = semantic_scores(
        semantic_model,
        [original_local] * len(raw_candidates),
        candidate_locals,
    )
    sentence_similarities = semantic_scores(
        semantic_model,
        [original_sentence] * len(raw_candidates),
        candidate_sentences,
    )

    rows: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        candidate_core = str(raw["candidate_core"])
        candidate_text = candidate_texts[index]
        words_after = candidate_words[index]
        candidate_sentence = candidate_sentences[index]
        candidate_local = candidate_locals[index]

        pos_ok, morphology_ok, original_pos, candidate_pos = pos_and_morphology_ok(
            pos_tagger,
            words_before,
            words_after,
            position,
            original_parts.core,
            candidate_core,
        )

        candidate_doc = score_document(model2, model4, candidate_text)
        candidate_attribution = full_token_attribution(model2, model4, candidate_text)
        verify_attribution(candidate_doc, candidate_attribution, candidate_text)

        anomaly_after = local_anomaly_sum(
            candidate_attribution, local_start, local_end, center, scale
        )
        delta_anomaly = anomaly_before - anomaly_after

        local_h4_after = kenlm_cross_entropy(model4, candidate_local)
        local_h4_gain = original_local_h4 - local_h4_after
        document_h4_change = float(candidate_doc["H4"] - current_doc["H4"])

        mlm_rank = int(raw["mlm_rank"])
        mlm_probability = float(raw["mlm_probability"])
        local_context_similarity = float(local_context_similarities[index])
        sentence_similarity = float(sentence_similarities[index])

        row: Dict[str, Any] = {
            "target_rank": int(target_row["target_rank"]),
            "position": position,
            "original_token": original_token,
            "original_core": original_parts.core,
            "candidate_core": candidate_core,
            "candidate_token": rebuild_token(original_parts, candidate_core),
            "candidate_source": str(raw.get("candidate_source", "")),
            "original_lemma": str(raw.get("original_lemma", rough_lemma(original_parts.core))),
            "candidate_lemma": str(raw.get("candidate_lemma", rough_lemma(candidate_core))),
            "wordnet_pos": str(raw.get("wordnet_pos", "")),
            "wordnet_shared_synsets": str(raw.get("wordnet_shared_synsets", "")),
            "wordnet_shared_synset_count": int(raw.get("wordnet_shared_synset_count", 0)),
            "wordnet_pool_size": int(raw.get("wordnet_pool_size", 0)),
            "wordnet_relation_ok": bool(raw.get("wordnet_relation_ok", False)),
            "frame_compatibility_ok": lexical_frame_compatibility_ok(
                str(raw.get("original_lemma", original_parts.core)),
                str(raw.get("candidate_lemma", candidate_core)),
                str(target_row.get("coarse_pos", "")),
            ),
            "target_d24": float(target_row["d24"]),
            "target_hrcg_anomaly": float(target_row["hrcg_anomaly"]),
            "target_score": float(target_row.get("target_score", target_row["hrcg_anomaly"])),
            "target_coarse_pos": str(target_row.get("coarse_pos", "")),
            "original_pos": original_pos or str(target_row.get("pos_tag", "")),
            "candidate_pos": candidate_pos,
            "mlm_rank": mlm_rank,
            "mlm_global_rank": int(raw.get("mlm_global_rank", 10**9)),
            "mlm_probability": mlm_probability,
            "mlm_log_probability": float(raw["mlm_log_probability"]),
            "mlm_context_ok": (
                mlm_rank <= int(max_mlm_rank)
                and int(raw.get("mlm_global_rank", 10**9)) <= int(max_mlm_global_rank)
                and mlm_probability >= float(min_mlm_probability)
            ),
            "local_context_similarity": local_context_similarity,
            "sentence_similarity": sentence_similarity,
            "local_context_semantic_ok": (
                local_context_similarity >= float(local_context_semantic_threshold)
            ),
            "sentence_semantic_ok": (
                sentence_similarity >= float(sentence_semantic_threshold)
            ),
            "pos_ok": bool(pos_ok),
            "morphology_ok": bool(morphology_ok),
            "numbers_preserved": (
                number_signature(original_sentence) == number_signature(candidate_sentence)
            ),
            "negation_preserved": (
                negation_signature(original_sentence) == negation_signature(candidate_sentence)
            ),
            "adjacent_duplication_ok": adjacent_duplication_ok(words_after, position),
            "adjacent_lemma_duplication_ok": adjacent_lemma_duplication_ok(words_after, position),
            "local_start": local_start,
            "local_end": local_end,
            "original_local": original_local,
            "candidate_local": candidate_local,
            "original_sentence": original_sentence,
            "candidate_sentence": candidate_sentence,
            "local_anomaly_before_analysis_only": anomaly_before,
            "local_anomaly_after_analysis_only": anomaly_after,
            "delta_local_hrcg_anomaly_analysis_only": delta_anomaly,
            "local_H4_before": original_local_h4,
            "local_H4_after": local_h4_after,
            "local_H4_gain": local_h4_gain,
            "local_h4_improved": local_h4_gain > float(min_local_h4_gain),
            "document_H4_before": float(current_doc["H4"]),
            "document_H4_after": float(candidate_doc["H4"]),
            "document_H4_change": document_h4_change,
            "document_h4_ok": (
                document_h4_change <= float(max_document_h4_increase)
            ),
            "A24_before": float(current_doc["A24"]),
            "A24_after": float(candidate_doc["A24"]),
            "delta_A24_analysis_only": float(current_doc["A24"] - candidate_doc["A24"]),
            "candidate_text": candidate_text,
        }
        preliminary_keys = [
            "wordnet_relation_ok",
            "frame_compatibility_ok",
            "mlm_context_ok",
            "local_context_semantic_ok",
            "sentence_semantic_ok",
            "pos_ok",
            "morphology_ok",
            "numbers_preserved",
            "negation_preserved",
            "adjacent_duplication_ok",
            "adjacent_lemma_duplication_ok",
            "local_h4_improved",
            "document_h4_ok",
        ]
        row["preliminary_valid_candidate"] = all(
            bool(row[key]) for key in preliminary_keys
        )
        row["semantic_validation_applied"] = False
        row["nli_forward_entailment"] = float("nan")
        row["nli_backward_entailment"] = float("nan")
        row["nli_min_bidirectional_entailment"] = float("nan")
        row["nli_forward_contradiction"] = float("nan")
        row["nli_backward_contradiction"] = float("nan")
        row["nli_max_bidirectional_contradiction"] = float("nan")
        row["nli_ok"] = False
        row["cola_original_acceptability"] = float("nan")
        row["cola_candidate_acceptability"] = float("nan")
        row["cola_acceptability_drop"] = float("nan")
        row["cola_ok"] = False
        row["semantic_equivalence_ok"] = False
        row["valid_candidate"] = False
        row["selected_for_round"] = False
        rows.append(row)

    validation_indices = [
        index
        for index, row in enumerate(rows)
        if bool(row["preliminary_valid_candidate"])
    ]
    if validation_indices:
        validation_results = semantic_gate.score(
            [rows[index]["original_sentence"] for index in validation_indices],
            [rows[index]["candidate_sentence"] for index in validation_indices],
        )
        for index, validation in zip(validation_indices, validation_results):
            rows[index].update(validation)
            rows[index]["semantic_validation_applied"] = True
            rows[index]["valid_candidate"] = bool(
                rows[index]["preliminary_valid_candidate"]
                and rows[index]["semantic_equivalence_ok"]
            )

    for row in rows:
        row["rejection_reason"] = "" if row["valid_candidate"] else rejection_reason(row)
    return rows


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def safe_name(raw_path: Path, source: str) -> str:
    name = raw_path.name.replace(".raw_data.json", "") + "_" + source
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)


def print_document_score(doc: Dict[str, float], title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"H2    : {doc['H2']:.10f}")
    print(f"H4    : {doc['H4']:.10f}")
    print(f"H2-H4 : {doc['H2'] - doc['H4']:.10f}")
    print(f"A24   : {doc['A24']:.10f}  (analysis only)")


def select_best_candidate(frame: pd.DataFrame) -> Optional[pd.Series]:
    valid = frame.loc[frame["valid_candidate"]].copy()
    if valid.empty:
        return None

    # Locked V2 ranking: human-reference local H4 gain is primary. HRCG has
    # already served its role by choosing the target location and is not used
    # again to select the replacement.
    return valid.sort_values(
        [
            "local_H4_gain",
            "mlm_log_probability",
            "local_context_similarity",
            "sentence_similarity",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def run_audit_or_rewrite(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(args.raw_file).expanduser() if args.raw_file else None
    if raw_path is None:
        files = sorted(RAW_DATA_DIR.glob("*.raw_data.json"))
        if not files:
            raise FileNotFoundError(f"No raw_data.json files in {RAW_DATA_DIR}")
        raw_path = files[0]
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    texts = extract_texts(raw_path, args.source)
    if not texts:
        raise RuntimeError(f"No {args.source} texts found in {raw_path}")
    if not (0 <= args.sample_index < len(texts)):
        raise IndexError(f"sample_index={args.sample_index}, available={len(texts)}")

    original_text = texts[args.sample_index]
    condition = safe_name(raw_path, args.source)

    print("=" * 100)
    print("GramGuard-V3 | HRCG-Guided WordNet-Synonym-Constrained Substitution")
    print("=" * 100)
    print(f"Mode        : {args.mode.upper()}")
    print(f"Input file  : {raw_path}")
    print(f"Source      : {args.source.upper()}")
    print(f"Sample index: {args.sample_index}")
    print(f"Words       : {word_count(original_text)}")
    print(f"Preview     : {original_text[:500]}{'...' if len(original_text) > 500 else ''}")

    print(f"\n[LOAD] UMBC 2-gram: {KENLM_2}")
    model2 = kenlm.Model(str(KENLM_2))
    print(f"[LOAD] UMBC 4-gram: {KENLM_4}")
    model4 = kenlm.Model(str(KENLM_4))

    mlm_path = resolve_mlm_path(args.mlm_model)
    device = choose_torch_device(args.device)
    tokenizer, masked_lm = load_masked_lm(mlm_path, device)

    print(f"\n[LOAD] Semantic model: {SEMANTIC_MODEL}")
    semantic_model = SentenceTransformer(
        str(SEMANTIC_MODEL),
        device="cpu",
        local_files_only=True,
    )

    validation_device = choose_torch_device(args.validation_device)
    nli_path = resolve_validation_model_path(
        args.nli_model,
        env_name="GRAMGUARD_NLI_MODEL",
        repo_cache_name="models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli",
        display_name="NLI model",
    )
    cola_path = resolve_validation_model_path(
        args.cola_model,
        env_name="GRAMGUARD_COLA_MODEL",
        repo_cache_name="models--textattack--roberta-base-CoLA",
        display_name="CoLA model",
    )
    semantic_gate = SemanticEquivalenceGate(
        nli_path=nli_path,
        cola_path=cola_path,
        device=validation_device,
        min_entailment=args.min_bidirectional_entailment,
        max_contradiction=args.max_bidirectional_contradiction,
        min_cola_acceptability=args.min_cola_acceptability,
        max_cola_drop=args.max_cola_drop,
    )
    print(
        "[GATE] NLI/CoLA thresholds: "
        f"entailment>={args.min_bidirectional_entailment:.3f}, "
        f"contradiction<={args.max_bidirectional_contradiction:.3f}, "
        f"CoLA>={args.min_cola_acceptability:.3f}, "
        f"CoLA drop<={args.max_cola_drop:.3f}"
    )

    pos_tagger = OptionalPosTagger()
    if not pos_tagger.available:
        raise RuntimeError(
            "Mandatory NLTK POS tagger/resource is unavailable. "
            f"Reason: {pos_tagger.reason}. Install it in the active environment with:\n"
            "  python - <<'PY'\n"
            "  import nltk\n"
            "  nltk.download('averaged_perceptron_tagger_eng')\n"
            "  nltk.download('averaged_perceptron_tagger')\n"
            "  PY"
        )
    print("[POS] NLTK POS tagger available: mandatory filtering enabled")

    synonym_provider = WordNetSynonymProvider()
    print("[LOAD] WordNet synonym resource: same-synset candidates only")

    working_text = original_text
    edited_positions: set[int] = set()
    all_target_rows: List[pd.DataFrame] = []
    all_candidate_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    stop_reason = "maximum edit budget reached"
    max_rounds = 1 if args.mode == "audit" else int(args.max_edits)

    initial_doc = score_document(model2, model4, original_text)
    print_document_score(initial_doc, "INITIAL DOCUMENT")

    for round_index in range(1, max_rounds + 1):
        current_doc = score_document(model2, model4, working_text)
        current_attr = full_token_attribution(model2, model4, working_text)
        verify_attribution(current_doc, current_attr, working_text)

        targets, center, scale = select_hrcg_targets(
            current_attr,
            working_text,
            pos_tagger,
            target_count=args.target_count,
            min_word_length=args.min_word_length,
            excluded_positions=edited_positions,
        )
        if targets.empty:
            stop_reason = "no editable HRCG target words"
            print(f"\n[STOP] {stop_reason}")
            break

        targets = targets.copy()
        targets.insert(0, "round", round_index)
        all_target_rows.append(targets)

        print("\n" + "#" * 100)
        print(f"ROUND {round_index}")
        print("#" * 100)
        print(f"Current A24 (analysis only): {current_doc['A24']:.10f}")
        print("\nHRCG TARGET WORDS")
        print(
            targets[
                [
                    "target_rank", "position", "token", "core", "pos_tag",
                    "coarse_pos", "token_frequency", "d24", "hrcg_anomaly",
                    "pos_editability_weight", "target_score",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
        )

        round_rows: List[Dict[str, Any]] = []
        for target in targets.itertuples(index=False):
            print(
                f"\n[ROUND {round_index} | TARGET {target.target_rank}] "
                f"position={target.position}, token={target.token!r}, anomaly={target.hrcg_anomaly:.4f}"
            )
            raw_candidates = generate_synonym_candidates(
                tokenizer,
                masked_lm,
                device,
                synonym_provider,
                working_text,
                int(target.position),
                penn_tag=str(target.pos_tag),
                coarse_pos=str(target.coarse_pos),
                top_k=args.mlm_candidates,
                max_context_words=args.max_context_words,
            )
            print(
                "WordNet same-synset candidates ranked by MLM: "
                f"{len(raw_candidates)}"
            )
            target_series = pd.Series(target._asdict())
            evaluated = evaluate_target_candidates(
                text=working_text,
                target_row=target_series,
                raw_candidates=raw_candidates,
                model2=model2,
                model4=model4,
                semantic_model=semantic_model,
                pos_tagger=pos_tagger,
                current_doc=current_doc,
                current_attribution=current_attr,
                center=center,
                scale=scale,
                local_radius=args.local_radius,
                local_context_semantic_threshold=args.local_context_semantic_threshold,
                sentence_semantic_threshold=args.sentence_semantic_threshold,
                max_mlm_rank=args.max_mlm_rank,
                max_mlm_global_rank=args.max_mlm_global_rank,
                min_mlm_probability=args.min_mlm_probability,
                min_local_h4_gain=args.min_local_h4_gain,
                max_document_h4_increase=args.max_document_h4_increase,
                semantic_gate=semantic_gate,
            )
            for row in evaluated:
                row["round"] = round_index
                round_rows.append(row)
                all_candidate_rows.append(row)

        if not round_rows:
            stop_reason = "WordNet/MLM produced no synonym-constrained candidates"
            print(f"\n[STOP] {stop_reason}")
            break

        round_df = pd.DataFrame(round_rows)
        round_df = round_df.sort_values(
            ["valid_candidate", "local_H4_gain", "mlm_log_probability", "local_context_similarity"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        print("\nLEXICAL CANDIDATE AUDIT")
        display_columns = [
            "target_rank", "original_core", "candidate_core", "candidate_source",
            "wordnet_shared_synset_count", "mlm_rank", "mlm_global_rank",
            "mlm_probability", "frame_compatibility_ok",
            "local_context_similarity", "sentence_similarity",
            "original_pos", "candidate_pos", "local_H4_gain", "document_H4_change",
            "nli_min_bidirectional_entailment", "nli_max_bidirectional_contradiction",
            "cola_candidate_acceptability", "cola_acceptability_drop",
            "delta_local_hrcg_anomaly_analysis_only", "valid_candidate", "rejection_reason",
        ]
        print(
            round_df[display_columns]
            .head(args.display_candidates)
            .to_string(index=False, float_format=lambda value: f"{value:.6f}")
        )

        best = select_best_candidate(round_df)
        if best is None:
            stop_reason = "no lexical candidate passed MLM, semantic, POS/morphology, and positive-H4 and semantic-equivalence safeguards"
            print(f"\n[STOP] {stop_reason}")
            break

        # Mark in both round-local and accumulated records.
        selected_mask = (
            (round_df["position"] == int(best["position"]))
            & (round_df["candidate_core"] == str(best["candidate_core"]))
        )
        round_df.loc[selected_mask, "selected_for_round"] = True
        for row in all_candidate_rows:
            if (
                int(row["round"]) == round_index
                and int(row["position"]) == int(best["position"])
                and str(row["candidate_core"]) == str(best["candidate_core"])
            ):
                row["selected_for_round"] = True

        selected_record = best.to_dict()
        selected_record["round"] = round_index
        selected_record["text_before"] = working_text
        selected_record["text_after"] = str(best["candidate_text"])
        selected_rows.append(selected_record)

        print("\nSELECTED LEXICAL EDIT")
        print(f"Position       : {int(best['position'])}")
        print(f"Original       : {best['original_token']}")
        print(f"Replacement    : {best['candidate_token']}")
        print(f"Local similarity: {best['local_context_similarity']:.6f}")
        print(f"Sent. similarity: {best['sentence_similarity']:.6f}")
        print(f"NLI min entail. : {best['nli_min_bidirectional_entailment']:.6f}")
        print(f"NLI max contra. : {best['nli_max_bidirectional_contradiction']:.6f}")
        print(f"CoLA accept.    : {best['cola_candidate_acceptability']:.6f}")
        print(f"CoLA drop       : {best['cola_acceptability_drop']:.6f}")
        print(
            f"WordNet synsets : {int(best['wordnet_shared_synset_count'])} "
            f"({best['wordnet_shared_synsets']})"
        )
        print(
            f"MLM pool/global: {int(best['mlm_rank'])} / "
            f"{int(best['mlm_global_rank'])}; prob={best['mlm_probability']:.8f}"
        )
        print(f"Local H4 gain  : {best['local_H4_gain']:.6f}")
        print(
            f"HRCG anomaly Δ : {best['delta_local_hrcg_anomaly_analysis_only']:.6f} "
            "(analysis only)"
        )
        print(f"Doc H4 change  : {best['document_H4_change']:.6f}")
        print(
            f"A24 change     : {best['delta_A24_analysis_only']:.6f} "
            "(analysis only; positive means decrease)"
        )
        print(f"Sentence before: {best['original_sentence']}")
        print(f"Sentence after : {best['candidate_sentence']}")

        if args.mode == "audit":
            stop_reason = "audit mode: candidate selected for inspection but not applied"
            break

        working_text = str(best["candidate_text"])
        edited_positions.add(int(best["position"]))

    final_doc = score_document(model2, model4, working_text)

    stem = f"{condition}_sample{args.sample_index}"
    targets_path = output_dir / f"lexical_targets_{stem}.csv"
    candidates_path = output_dir / f"lexical_candidates_{stem}.csv"
    selected_path = output_dir / f"lexical_selected_{stem}.csv"
    final_path = output_dir / f"lexical_final_{stem}.txt"
    summary_path = output_dir / f"lexical_summary_{stem}.csv"

    targets_df = pd.concat(all_target_rows, ignore_index=True) if all_target_rows else pd.DataFrame()
    candidates_df = pd.DataFrame(all_candidate_rows)
    selected_df = pd.DataFrame(selected_rows)

    targets_df.to_csv(targets_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    final_path.write_text(working_text, encoding="utf-8")

    valid_count = int(candidates_df["valid_candidate"].sum()) if not candidates_df.empty else 0
    summary = {
        "mode": args.mode,
        "input_file": str(raw_path),
        "source": args.source,
        "sample_index": int(args.sample_index),
        "mlm_model": str(mlm_path),
        "pos_tagger_available": bool(pos_tagger.available),
        "locked_target_policy": "HRCG_x_editability_prior",
        "locked_candidate_pool": "wordnet_same_synset_single_token_only",
        "locked_candidate_objective": "positive_local_H4_gain_then_MLM_probability",
        "lexical_frame_guard": "complement_sensitive_verbs_blocked",
        "semantic_validation": "bidirectional_NLI_plus_CoLA_plus_adjacent_lemma_guard",
        "max_mlm_global_rank": int(args.max_mlm_global_rank),
        "nli_model": str(nli_path),
        "cola_model": str(cola_path),
        "min_bidirectional_entailment": float(args.min_bidirectional_entailment),
        "max_bidirectional_contradiction": float(args.max_bidirectional_contradiction),
        "min_cola_acceptability": float(args.min_cola_acceptability),
        "max_cola_drop": float(args.max_cola_drop),
        "target_rows": int(len(targets_df)),
        "candidate_count": int(len(candidates_df)),
        "valid_candidate_count": valid_count,
        "selected_edit_count": int(len(selected_df)),
        "applied_edit_count": int(len(selected_df)) if args.mode == "rewrite" else 0,
        "initial_H2": float(initial_doc["H2"]),
        "initial_H4": float(initial_doc["H4"]),
        "initial_A24": float(initial_doc["A24"]),
        "final_H2": float(final_doc["H2"]),
        "final_H4": float(final_doc["H4"]),
        "final_A24": float(final_doc["A24"]),
        "stop_reason": stop_reason,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("GRAMGUARD-V3 WORDNET-SYNONYM RESULT")
    print("=" * 100)
    print(f"Mode              : {args.mode}")
    print(f"Candidates        : {len(candidates_df)}")
    print(f"Valid candidates  : {valid_count}")
    print(f"Selected edits    : {len(selected_df)}")
    print(f"Applied edits     : {summary['applied_edit_count']}")
    print(f"Initial H4        : {initial_doc['H4']:.10f}")
    print(f"Final H4          : {final_doc['H4']:.10f}")
    print(f"Initial A24       : {initial_doc['A24']:.10f}")
    print(f"Final A24         : {final_doc['A24']:.10f}")
    print(f"Stop reason       : {stop_reason}")
    print(f"Saved targets     : {targets_path}")
    print(f"Saved candidates  : {candidates_path}")
    print(f"Saved selected    : {selected_path}")
    print(f"Saved final text  : {final_path}")
    print(f"Saved summary     : {summary_path}")



# -----------------------------------------------------------------------------
# Paired attack evaluation
# -----------------------------------------------------------------------------


def parse_index_spec(spec: str) -> List[int]:
    """Parse comma/range syntax such as '0-4,7,10-12'."""
    values: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"Invalid descending index range: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    if not values:
        raise ValueError("--eval_indices resolved to an empty set")
    return sorted(values)


def _has_causal_weights(path: Path) -> bool:
    patterns = [
        "model.safetensors",
        "pytorch_model.bin",
        "model-*.safetensors",
        "pytorch_model-*.bin",
    ]
    return any(any(path.glob(pattern)) for pattern in patterns)


def _has_causal_tokenizer(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ]
    )


def discover_local_gptneo_snapshots() -> List[Path]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    roots = sorted(hub.glob("models--EleutherAI--gpt-neo-2.7B/snapshots/*"))
    usable = [
        path
        for path in roots
        if path.is_dir()
        and (path / "config.json").exists()
        and _has_causal_weights(path)
        and _has_causal_tokenizer(path)
    ]
    return usable


def resolve_causal_model_path(spec: str) -> Path:
    requested = str(spec).strip()
    if requested and requested.lower() != "auto":
        path = Path(os.path.expanduser(requested)).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Causal model path does not exist: {path}")
        return path

    env_path = os.environ.get("GRAMGUARD_CAUSAL_MODEL", "").strip()
    if env_path:
        path = Path(os.path.expanduser(env_path)).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"GRAMGUARD_CAUSAL_MODEL points to a missing path: {path}"
            )
        return path

    candidates = discover_local_gptneo_snapshots()
    if not candidates:
        raise FileNotFoundError(
            "No usable local EleutherAI/gpt-neo-2.7B snapshot was found. "
            "Provide --causal_model /absolute/path/to/snapshot."
        )
    # Snapshot hashes are immutable; the most recently modified usable snapshot
    # is preferred when multiple local copies exist.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_causal_scorer(
    model_path: Path,
    device: str,
) -> Tuple[Any, Any]:
    print(f"[LOAD] Causal tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[LOAD] Causal model    : {model_path}")
    dtype = torch.float32
    kwargs = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        # Compatibility with older transformers versions.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            **kwargs,
        )
    model.to(device)
    model.eval()
    print(f"[LOAD] Causal device   : {device}")
    return tokenizer, model


def score_causal_text(
    text: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_length: int,
) -> Dict[str, float]:
    """
    Compute four fixed-orientation zero-shot scores in one forward pass.

    Every returned detector score is oriented so that a larger value means
    "more machine-like":

      likelihood_higher_ai = mean observed next-token log probability
      entropy_higher_ai    = negative mean predictive entropy
      logrank_higher_ai    = negative mean log rank of observed tokens
      fastdetect_higher_ai = analytic sampling-discrepancy z score

    Fast-DetectGPT uses the same local GPT-Neo-2.7B model as both reference and
    scoring model, matching the project's established analytic configuration.
    """
    cleaned = clean_text(text)
    encoded = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if input_ids.shape[1] < 2:
        raise ValueError("Text has fewer than two causal-LM tokens")

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        observed = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Predictive entropy and analytic Fast-DetectGPT moments.
        expected_logp = torch.sum(probs * log_probs, dim=-1)
        expected_logp2 = torch.sum(probs * log_probs.square(), dim=-1)
        variance = torch.clamp(
            expected_logp2 - expected_logp.square(),
            min=1e-12,
        )
        entropy = -expected_logp

        # Exact observed-token rank; ties are assigned the best tied rank.
        target_logits = logits.gather(-1, targets.unsqueeze(-1))
        ranks = 1 + torch.sum(logits > target_logits, dim=-1)
        log_ranks = torch.log(ranks.float())

        valid_mask = torch.ones_like(observed, dtype=torch.bool)
        if attention_mask is not None:
            valid_mask = attention_mask[:, 1:].bool()

        observed = observed[valid_mask]
        expected_logp = expected_logp[valid_mask]
        variance = variance[valid_mask]
        entropy = entropy[valid_mask]
        log_ranks = log_ranks[valid_mask]

        token_count = int(observed.numel())
        if token_count == 0:
            raise ValueError("No valid causal-LM scoring positions")

        likelihood = observed.mean()
        entropy_mean = entropy.mean()
        logrank_mean = log_ranks.mean()
        discrepancy = (
            (observed.sum() - expected_logp.sum())
            / torch.sqrt(variance.sum().clamp_min(1e-12))
        )

    return {
        "causal_token_count": token_count,
        "likelihood_higher_ai": float(likelihood.cpu()),
        "entropy_higher_ai": float((-entropy_mean).cpu()),
        "logrank_higher_ai": float((-logrank_mean).cpu()),
        "fastdetect_higher_ai": float(discrepancy.cpu()),
        "mean_log_probability": float(likelihood.cpu()),
        "mean_predictive_entropy": float(entropy_mean.cpu()),
        "mean_log_rank": float(logrank_mean.cpu()),
    }


def load_applied_edit_count(
    attack_output_dir: Path,
    condition: str,
    sample_index: int,
) -> int:
    summary_path = (
        attack_output_dir
        / f"lexical_summary_{condition}_sample{sample_index}.csv"
    )
    if not summary_path.exists():
        return 0
    frame = pd.read_csv(summary_path)
    if frame.empty or "applied_edit_count" not in frame.columns:
        return 0
    return int(frame.iloc[0]["applied_edit_count"])


def load_selected_edit_description(
    attack_output_dir: Path,
    condition: str,
    sample_index: int,
) -> str:
    selected_path = (
        attack_output_dir
        / f"lexical_selected_{condition}_sample{sample_index}.csv"
    )
    if not selected_path.exists() or selected_path.stat().st_size == 0:
        return ""
    try:
        frame = pd.read_csv(selected_path)
    except pd.errors.EmptyDataError:
        # A document with no accepted edit legitimately has an empty selected CSV.
        return ""
    if frame.empty:
        return ""

    pieces: List[str] = []
    for row in frame.itertuples(index=False):
        original = getattr(row, "original_core", getattr(row, "original_token", ""))
        candidate = getattr(row, "candidate_core", getattr(row, "candidate_token", ""))
        if str(original) and str(candidate):
            pieces.append(f"{original}->{candidate}")
    return " | ".join(pieces)


def save_incremental_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run_paired_evaluation(args: argparse.Namespace) -> None:
    raw_path = Path(args.raw_file).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    attack_output_dir = Path(args.attack_output_dir).expanduser().resolve()
    if not attack_output_dir.exists():
        raise FileNotFoundError(
            f"Attack output directory does not exist: {attack_output_dir}"
        )

    output_dir = Path(args.eval_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    originals = extract_texts(raw_path, args.source)
    if not originals:
        raise RuntimeError(f"No {args.source} texts found in {raw_path}")

    indices = parse_index_spec(args.eval_indices)
    invalid = [index for index in indices if not 0 <= index < len(originals)]
    if invalid:
        raise IndexError(
            f"Evaluation indices outside available range 0..{len(originals)-1}: {invalid}"
        )

    condition = safe_name(raw_path, args.source)
    pairs_jsonl = output_dir / f"attack_pairs_{condition}.jsonl"
    scores_csv = output_dir / f"attack_detector_scores_{condition}.csv"
    deltas_csv = output_dir / f"attack_detector_deltas_{condition}.csv"
    summary_csv = output_dir / f"attack_detector_summary_{condition}.csv"

    print("=" * 100)
    print("GramGuard-V3 | Paired Original-vs-Attacked Detector Evaluation")
    print("=" * 100)
    print(f"Input file       : {raw_path}")
    print(f"Attack outputs   : {attack_output_dir}")
    print(f"Indices          : {indices}")
    print(f"Maximum LM length: {args.eval_max_length}")
    print("Score orientation: every *_higher_ai metric uses larger = more AI")
    print("Attack success   : attacked score - original score < 0")

    print(f"\n[LOAD] UMBC 2-gram: {KENLM_2}")
    model2 = kenlm.Model(str(KENLM_2))
    print(f"[LOAD] UMBC 4-gram: {KENLM_4}")
    model4 = kenlm.Model(str(KENLM_4))

    causal_path = resolve_causal_model_path(args.causal_model)
    causal_device = choose_torch_device(args.causal_device)
    tokenizer, causal_model = load_causal_scorer(
        causal_path,
        causal_device,
    )

    pair_records: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []

    # Resume safely from incremental files created by an interrupted run.
    # Keep only complete original/attacked pairs for the requested indices.
    if scores_csv.exists() and scores_csv.stat().st_size > 0:
        try:
            previous_scores = pd.read_csv(scores_csv)
        except pd.errors.EmptyDataError:
            previous_scores = pd.DataFrame()
        required = {"sample_index", "variant"}
        if not previous_scores.empty and required.issubset(previous_scores.columns):
            previous_scores = previous_scores[
                previous_scores["sample_index"].isin(indices)
            ].copy()
            previous_scores = previous_scores.drop_duplicates(
                subset=["sample_index", "variant"], keep="last"
            )
            score_rows = previous_scores.to_dict(orient="records")

    if pairs_jsonl.exists() and pairs_jsonl.stat().st_size > 0:
        loaded_pairs: Dict[int, Dict[str, Any]] = {}
        with pairs_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                index = int(row.get("sample_index", -1))
                if index in indices:
                    loaded_pairs[index] = row
        pair_records = [loaded_pairs[index] for index in indices if index in loaded_pairs]

    completed_indices = {
        index
        for index in indices
        if {
            str(row.get("variant"))
            for row in score_rows
            if int(row.get("sample_index", -1)) == index
        } >= {"original", "attacked"}
    }
    if completed_indices:
        print(f"[RESUME] Complete cached pairs: {sorted(completed_indices)}")

    for number, sample_index in enumerate(indices, start=1):
        if sample_index in completed_indices:
            print(
                f"\n[{number}/{len(indices)}] sample={sample_index} "
                "already complete; reusing cached scores"
            )
            continue
        original_text = originals[sample_index]
        final_path = (
            attack_output_dir
            / f"lexical_final_{condition}_sample{sample_index}.txt"
        )
        if not final_path.exists():
            raise FileNotFoundError(
                f"Missing attacked final text for sample {sample_index}: {final_path}"
            )
        attacked_text = clean_text(
            final_path.read_text(encoding="utf-8", errors="replace")
        )
        applied_edits = load_applied_edit_count(
            attack_output_dir,
            condition,
            sample_index,
        )
        edit_description = load_selected_edit_description(
            attack_output_dir,
            condition,
            sample_index,
        )
        text_changed = attacked_text != original_text

        if bool(applied_edits) != bool(text_changed):
            print(
                f"[WARN] sample {sample_index}: applied_edit_count={applied_edits} "
                f"but text_changed={text_changed}"
            )

        pair_record = {
            "sample_index": sample_index,
            "source": args.source,
            "generator_condition": condition,
            "applied_edit_count": applied_edits,
            "edit_description": edit_description,
            "text_changed": text_changed,
            "original_text": original_text,
            "attacked_text": attacked_text,
        }
        pair_records.append(pair_record)

        print("\n" + "-" * 100)
        print(
            f"[{number}/{len(indices)}] sample={sample_index} "
            f"edits={applied_edits} changed={text_changed} "
            f"{edit_description}"
        )

        original_hr = score_document(model2, model4, original_text)
        original_causal = score_causal_text(
            original_text,
            tokenizer,
            causal_model,
            causal_device,
            args.eval_max_length,
        )
        original_row = {
            "sample_index": sample_index,
            "variant": "original",
            "applied_edit_count": applied_edits,
            "edit_description": edit_description,
            "text_changed": text_changed,
            "word_count": word_count(original_text),
            "H2": float(original_hr["H2"]),
            "H4": float(original_hr["H4"]),
            "A24_analysis_only": float(original_hr["A24"]),
            **original_causal,
        }
        score_rows.append(original_row)
        print(
            "  original "
            f"FastDetect={original_row['fastdetect_higher_ai']:.6f} "
            f"Likelihood={original_row['likelihood_higher_ai']:.6f} "
            f"EntropyAI={original_row['entropy_higher_ai']:.6f} "
            f"LogRankAI={original_row['logrank_higher_ai']:.6f}"
        )

        if text_changed:
            attacked_hr = score_document(model2, model4, attacked_text)
            attacked_causal = score_causal_text(
                attacked_text,
                tokenizer,
                causal_model,
                causal_device,
                args.eval_max_length,
            )
        else:
            # Identical text must have exactly identical scores; avoid a duplicate
            # GPT-Neo forward pass and copy the original results.
            attacked_hr = dict(original_hr)
            attacked_causal = dict(original_causal)

        attacked_row = {
            "sample_index": sample_index,
            "variant": "attacked",
            "applied_edit_count": applied_edits,
            "edit_description": edit_description,
            "text_changed": text_changed,
            "word_count": word_count(attacked_text),
            "H2": float(attacked_hr["H2"]),
            "H4": float(attacked_hr["H4"]),
            "A24_analysis_only": float(attacked_hr["A24"]),
            **attacked_causal,
        }
        score_rows.append(attacked_row)
        print(
            "  attacked "
            f"FastDetect={attacked_row['fastdetect_higher_ai']:.6f} "
            f"Likelihood={attacked_row['likelihood_higher_ai']:.6f} "
            f"EntropyAI={attacked_row['entropy_higher_ai']:.6f} "
            f"LogRankAI={attacked_row['logrank_higher_ai']:.6f}"
        )

        # Preserve useful partial results if a later text fails.
        save_incremental_csv(score_rows, scores_csv)
        with pairs_jsonl.open("w", encoding="utf-8") as handle:
            for row in pair_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Normalize resumed and newly computed rows before aggregation.
    score_frame = pd.DataFrame(score_rows)
    score_frame = score_frame.drop_duplicates(
        subset=["sample_index", "variant"], keep="last"
    )
    score_frame = score_frame[score_frame["sample_index"].isin(indices)].copy()
    score_frame = score_frame.sort_values(["sample_index", "variant"]).reset_index(drop=True)
    score_frame.to_csv(scores_csv, index=False)

    pair_map = {int(row["sample_index"]): row for row in pair_records}
    pair_records = [pair_map[index] for index in indices if index in pair_map]
    with pairs_jsonl.open("w", encoding="utf-8") as handle:
        for row in pair_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    missing_pairs = [
        index
        for index in indices
        if not (
            ((score_frame["sample_index"] == index) & (score_frame["variant"] == "original")).any()
            and ((score_frame["sample_index"] == index) & (score_frame["variant"] == "attacked")).any()
        )
    ]
    if missing_pairs:
        raise RuntimeError(f"Evaluation is incomplete for samples: {missing_pairs}")
    metric_columns = [
        "likelihood_higher_ai",
        "entropy_higher_ai",
        "logrank_higher_ai",
        "fastdetect_higher_ai",
    ]
    diagnostic_columns = ["H2", "H4", "A24_analysis_only"]

    delta_rows: List[Dict[str, Any]] = []
    for sample_index in indices:
        original = score_frame[
            (score_frame["sample_index"] == sample_index)
            & (score_frame["variant"] == "original")
        ].iloc[0]
        attacked = score_frame[
            (score_frame["sample_index"] == sample_index)
            & (score_frame["variant"] == "attacked")
        ].iloc[0]

        row: Dict[str, Any] = {
            "sample_index": sample_index,
            "applied_edit_count": int(attacked["applied_edit_count"]),
            "edit_description": attacked["edit_description"],
            "text_changed": bool(attacked["text_changed"]),
        }
        for metric in metric_columns:
            delta = float(attacked[metric] - original[metric])
            row[f"original_{metric}"] = float(original[metric])
            row[f"attacked_{metric}"] = float(attacked[metric])
            row[f"delta_{metric}"] = delta
            row[f"success_{metric}"] = bool(delta < 0.0) if row["text_changed"] else False
            row[f"delta_per_edit_{metric}"] = (
                delta / row["applied_edit_count"]
                if row["applied_edit_count"] > 0
                else 0.0
            )
        for metric in diagnostic_columns:
            row[f"original_{metric}"] = float(original[metric])
            row[f"attacked_{metric}"] = float(attacked[metric])
            row[f"delta_{metric}"] = float(attacked[metric] - original[metric])
        delta_rows.append(row)

    delta_frame = pd.DataFrame(delta_rows)
    delta_frame.to_csv(deltas_csv, index=False)

    edited = delta_frame[delta_frame["text_changed"]].copy()
    summary_rows: List[Dict[str, Any]] = []
    for metric in metric_columns:
        deltas = edited[f"delta_{metric}"] if not edited.empty else pd.Series(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "orientation": "higher=AI",
                "edited_documents": int(len(edited)),
                "successful_documents": int((deltas < 0).sum()) if len(deltas) else 0,
                "success_rate": float((deltas < 0).mean()) if len(deltas) else float("nan"),
                "mean_delta_attacked_minus_original": float(deltas.mean()) if len(deltas) else float("nan"),
                "median_delta_attacked_minus_original": float(deltas.median()) if len(deltas) else float("nan"),
                "mean_delta_per_edit": float(
                    edited[f"delta_per_edit_{metric}"].mean()
                ) if len(edited) else float("nan"),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(summary_csv, index=False)

    print("\n" + "=" * 100)
    print("PAIRED ATTACK EVALUATION SUMMARY")
    print("=" * 100)
    if edited.empty:
        print("No edited documents were found.")
    else:
        print(
            summary_frame.to_string(
                index=False,
                float_format=lambda value: f"{value:.8f}",
            )
        )
    print(f"\nSaved paired texts : {pairs_jsonl}")
    print(f"Saved raw scores   : {scores_csv}")
    print(f"Saved paired deltas: {deltas_csv}")
    print(f"Saved summary      : {summary_csv}")


# -----------------------------------------------------------------------------
# HRCG-guided masked-span reconstruction
# -----------------------------------------------------------------------------


def _eligible_span_positions(
    attribution: pd.DataFrame,
    text: str,
    pos_tagger: OptionalPosTagger,
    min_word_length: int,
    excluded_positions: Optional[set[int]] = None,
) -> pd.DataFrame:
    words = clean_text(text).split()
    tags = pos_tagger.tag_words(words)
    textual = attribution.loc[~attribution["is_eos"]].copy()
    center, scale = robust_center_scale(textual["d24"].to_numpy(dtype=float))
    textual["hrcg_center"] = center
    textual["hrcg_scale"] = scale
    textual["hrcg_anomaly"] = np.abs(textual["d24"] - center) / scale

    freq: Dict[str, int] = {}
    for word in words:
        core = split_token_parts(word).core.lower()
        if core:
            freq[core] = freq.get(core, 0) + 1

    excluded_positions = excluded_positions or set()
    rows: List[Dict[str, Any]] = []
    for row in textual.itertuples(index=False):
        position = int(row.position)
        core = str(row.core)
        tag = tags[position]
        coarse = pos_tagger.coarse(tag)
        if position in excluded_positions:
            ok, reason = False, "already_edited_or_nearby"
        else:
            ok, reason = lexical_target_ok(
                words, position, core, tag, pos_tagger, min_word_length, freq
            )
        record = row._asdict()
        record.update({
            "pos_tag": tag,
            "coarse_pos": coarse,
            "token_frequency": int(freq.get(core.lower(), 0)),
            "pos_editability_weight": float(POS_EDITABILITY_WEIGHT.get(coarse, 0.0)),
            "editable": bool(ok),
            "target_exclusion_reason": reason,
        })
        rows.append(record)
    return pd.DataFrame(rows)


def select_span_targets(
    attribution: pd.DataFrame,
    text: str,
    pos_tagger: OptionalPosTagger,
    selector: str,
    target_count: int,
    min_word_length: int,
    excluded_positions: Optional[set[int]],
    rng: random.Random,
) -> Tuple[pd.DataFrame, float, float]:
    frame = _eligible_span_positions(
        attribution, text, pos_tagger, min_word_length, excluded_positions
    )
    center = float(frame["hrcg_center"].iloc[0]) if not frame.empty else 0.0
    scale = float(frame["hrcg_scale"].iloc[0]) if not frame.empty else 1.0
    editable = frame.loc[frame["editable"]].copy()
    if editable.empty:
        return editable, center, scale

    if selector == "hrcg":
        editable["target_score"] = (
            editable["hrcg_anomaly"] * editable["pos_editability_weight"]
        )
        selected = editable.sort_values(
            ["target_score", "hrcg_anomaly", "abs_d24"],
            ascending=[False, False, False],
        ).head(int(target_count))
    elif selector == "h4":
        # Baseline localization: ordinary human-reference 4-gram surprisal only.
        editable["target_score"] = editable["ell4"] * editable["pos_editability_weight"]
        selected = editable.sort_values(
            ["target_score", "ell4"], ascending=[False, False]
        ).head(int(target_count))
    elif selector == "random":
        indices = list(editable.index)
        rng.shuffle(indices)
        selected = editable.loc[indices[: int(target_count)]].copy()
        selected["target_score"] = 0.0
    else:
        raise ValueError(f"Unknown span selector: {selector}")

    selected = selected.reset_index(drop=True)
    selected.insert(0, "target_rank", range(1, len(selected) + 1))
    selected["span_selector"] = selector
    return selected, center, scale


def _span_bounds(words: Sequence[str], center_position: int, radius: int) -> Tuple[int, int]:
    sent_start, sent_end = sentence_bounds(words, center_position)
    start = max(sent_start, center_position - int(radius))
    end = min(sent_end, center_position + int(radius))
    return start, end


def _span_mask_positions(
    words: Sequence[str],
    start: int,
    end: int,
    pos_tagger: OptionalPosTagger,
    center_position: int,
) -> List[int]:
    tags = pos_tagger.tag_words(words)
    positions: List[int] = []
    # Always attempt the HRCG/H4/random-selected center first.
    ordered = [center_position] + [p for p in range(start, end + 1) if p != center_position]
    for position in ordered:
        parts = split_token_parts(words[position])
        if not parts.core or not WORD_CORE_RE.fullmatch(parts.core):
            continue
        lower = parts.core.lower()
        if lower in STOPWORDS or lower in AUXILIARY_LEMMAS or any(ch.isdigit() for ch in lower):
            continue
        if pos_tagger.coarse(tags[position]) not in CONTENT_POS:
            continue
        positions.append(position)
        if len(positions) >= 3:
            break
    return positions


def _mlm_alternatives_at_position(
    text: str,
    position: int,
    tokenizer,
    masked_lm,
    device: torch.device,
    branch_k: int,
    max_context_words: int,
) -> List[Dict[str, Any]]:
    words = clean_text(text).split()
    original = split_token_parts(words[position]).core
    masked_context, _, _ = build_masked_context(
        text, position, tokenizer.mask_token, max_context_words
    )
    encoded = tokenizer(
        masked_context,
        return_tensors="pt",
        truncation=True,
        max_length=min(512, int(getattr(tokenizer, "model_max_length", 512) or 512)),
    )
    input_ids = encoded["input_ids"]
    mask_locations = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False).flatten()
    if len(mask_locations) != 1:
        return []
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = masked_lm(**encoded).logits[0, int(mask_locations[0])]
        probs = torch.softmax(logits, dim=-1)
    search_k = min(int(probs.numel()), max(80, int(branch_k) * 20))
    values, ids = torch.topk(probs, k=search_k)

    alternatives: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rank, (prob, token_id) in enumerate(zip(values.tolist(), ids.tolist()), start=1):
        token = normalize_decoded_token(tokenizer, int(token_id))
        if not token or token.startswith("##") or not WORD_CORE_RE.fullmatch(token):
            continue
        surface = preserve_case(token, original)
        lower = surface.lower()
        if lower in seen:
            continue
        seen.add(lower)
        alternatives.append({
            "surface": surface,
            "probability": float(prob),
            "log_probability": float(math.log(max(float(prob), 1e-30))),
            "global_rank": rank,
        })
        if len(alternatives) >= int(branch_k):
            break
    return alternatives


def generate_masked_span_candidates(
    *,
    text: str,
    center_position: int,
    span_radius: int,
    tokenizer,
    masked_lm,
    device: torch.device,
    pos_tagger: OptionalPosTagger,
    branch_k: int,
    beam_width: int,
    candidate_count: int,
    max_context_words: int,
) -> Tuple[List[Dict[str, Any]], int, int, List[int]]:
    words = clean_text(text).split()
    start, end = _span_bounds(words, center_position, span_radius)
    mask_positions = _span_mask_positions(words, start, end, pos_tagger, center_position)
    if not mask_positions:
        return [], start, end, []

    beams: List[Dict[str, Any]] = [{
        "text": text,
        "log_probability": 0.0,
        "replacement_map": {},
    }]

    for position in mask_positions:
        expanded: List[Dict[str, Any]] = []
        for beam in beams:
            alternatives = _mlm_alternatives_at_position(
                beam["text"], position, tokenizer, masked_lm, device,
                branch_k, max_context_words,
            )
            # Retain the original token as an explicit no-change branch. This lets
            # a span reconstruction alter one, two, or three content words rather
            # than forcing semantically destructive changes at every position.
            beam_words = clean_text(beam["text"]).split()
            original_core = split_token_parts(beam_words[position]).core
            if original_core and all(a["surface"].lower() != original_core.lower() for a in alternatives):
                alternatives.append({
                    "surface": original_core,
                    "probability": 1e-8,
                    "log_probability": math.log(1e-8),
                    "global_rank": 10**9,
                })

            for alt in alternatives:
                candidate_text = replace_word_core(beam["text"], position, alt["surface"])
                repl = dict(beam["replacement_map"])
                repl[position] = alt["surface"]
                expanded.append({
                    "text": candidate_text,
                    "log_probability": float(beam["log_probability"] + alt["log_probability"]),
                    "replacement_map": repl,
                })
        expanded.sort(key=lambda item: item["log_probability"], reverse=True)
        # Deduplicate complete intermediate texts before truncating the beam.
        unique: List[Dict[str, Any]] = []
        seen_texts: set[str] = set()
        for item in expanded:
            if item["text"] in seen_texts:
                continue
            seen_texts.add(item["text"])
            unique.append(item)
            if len(unique) >= int(beam_width):
                break
        beams = unique
        if not beams:
            break

    original_words = clean_text(text).split()
    results: List[Dict[str, Any]] = []
    for beam in beams:
        candidate_words = clean_text(beam["text"]).split()
        changed = [
            p for p in mask_positions
            if split_token_parts(original_words[p]).core.lower()
            != split_token_parts(candidate_words[p]).core.lower()
        ]
        if not changed:
            continue
        results.append({
            "candidate_text": beam["text"],
            "mlm_log_probability": float(beam["log_probability"]),
            "changed_positions": changed,
            "changed_token_count": len(changed),
            "replacement_map": beam["replacement_map"],
        })
    results.sort(key=lambda item: item["mlm_log_probability"], reverse=True)
    return results[: int(candidate_count)], start, end, mask_positions


def _span_pos_agreement(
    pos_tagger: OptionalPosTagger,
    before_words: Sequence[str],
    after_words: Sequence[str],
    changed_positions: Sequence[int],
) -> float:
    if not changed_positions:
        return 1.0
    before_tags = pos_tagger.tag_words(before_words)
    after_tags = pos_tagger.tag_words(after_words)
    agreements = [
        pos_tagger.coarse(before_tags[p]) == pos_tagger.coarse(after_tags[p])
        for p in changed_positions
    ]
    return float(sum(bool(x) for x in agreements) / len(agreements))


def evaluate_span_candidates(
    *,
    text: str,
    target_row: pd.Series,
    raw_candidates: List[Dict[str, Any]],
    span_start: int,
    span_end: int,
    model2: kenlm.Model,
    model4: kenlm.Model,
    semantic_model: SentenceTransformer,
    semantic_gate: SemanticEquivalenceGate,
    pos_tagger: OptionalPosTagger,
    current_doc: Dict[str, float],
    local_radius: int,
    min_local_similarity: float,
    min_sentence_similarity: float,
    min_pos_agreement: float,
    min_mlm_probability: float,
    min_local_h4_gain: float,
    max_document_h4_increase: float,
    max_edit_ratio: float,
    total_changed_positions: set[int],
) -> List[Dict[str, Any]]:
    if not raw_candidates:
        return []
    words_before = clean_text(text).split()
    center_position = int(target_row["position"])
    sent_start, sent_end = sentence_bounds(words_before, center_position)
    local_start = max(0, span_start - int(local_radius))
    local_end = min(len(words_before) - 1, span_end + int(local_radius))
    original_sentence = slice_words(words_before, sent_start, sent_end)
    original_local = slice_words(words_before, local_start, local_end)
    original_span = slice_words(words_before, span_start, span_end)
    original_local_h4 = kenlm_cross_entropy(model4, original_local)

    candidate_sentences: List[str] = []
    candidate_locals: List[str] = []
    for raw in raw_candidates:
        words_after = clean_text(raw["candidate_text"]).split()
        candidate_sentences.append(slice_words(words_after, sent_start, sent_end))
        candidate_locals.append(slice_words(words_after, local_start, local_end))

    local_sims = semantic_scores(
        semantic_model, [original_local] * len(raw_candidates), candidate_locals
    )
    sentence_sims = semantic_scores(
        semantic_model, [original_sentence] * len(raw_candidates), candidate_sentences
    )

    rows: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_candidates):
        candidate_text = clean_text(raw["candidate_text"])
        words_after = candidate_text.split()
        changed_positions = list(raw["changed_positions"])
        candidate_sentence = candidate_sentences[i]
        candidate_local = candidate_locals[i]
        candidate_span = slice_words(words_after, span_start, span_end)
        local_h4_after = kenlm_cross_entropy(model4, candidate_local)
        local_h4_gain = float(original_local_h4 - local_h4_after)
        candidate_doc = score_document(model2, model4, candidate_text)
        document_h4_change = float(candidate_doc["H4"] - current_doc["H4"])
        pos_agreement = _span_pos_agreement(
            pos_tagger, words_before, words_after, changed_positions
        )
        cumulative_changed = set(total_changed_positions) | set(changed_positions)
        edit_ratio = len(cumulative_changed) / max(1, len(words_before))
        local_similarity = float(local_sims[i])
        sentence_similarity = float(sentence_sims[i])
        avg_probability = float(math.exp(raw["mlm_log_probability"] / max(1, len(raw["replacement_map"]))))

        row: Dict[str, Any] = {
            "target_rank": int(target_row["target_rank"]),
            "span_selector": str(target_row.get("span_selector", "")),
            "position": center_position,
            "span_start": int(span_start),
            "span_end": int(span_end),
            "original_core": original_span,
            "candidate_core": candidate_span,
            "original_span": original_span,
            "candidate_span": candidate_span,
            "changed_positions": "|".join(map(str, changed_positions)),
            "changed_token_count": int(len(changed_positions)),
            "cumulative_changed_token_count": int(len(cumulative_changed)),
            "actual_edit_ratio": float(edit_ratio),
            "mlm_log_probability": float(raw["mlm_log_probability"]),
            "mlm_geomean_probability": avg_probability,
            "local_context_similarity": local_similarity,
            "sentence_similarity": sentence_similarity,
            "span_pos_agreement": pos_agreement,
            "numbers_preserved": number_signature(original_sentence) == number_signature(candidate_sentence),
            "negation_preserved": negation_signature(original_sentence) == negation_signature(candidate_sentence),
            "local_H4_before": float(original_local_h4),
            "local_H4_after": float(local_h4_after),
            "local_H4_gain": local_h4_gain,
            "document_H4_before": float(current_doc["H4"]),
            "document_H4_after": float(candidate_doc["H4"]),
            "document_H4_change": document_h4_change,
            "A24_before": float(current_doc["A24"]),
            "A24_after": float(candidate_doc["A24"]),
            "target_d24": float(target_row.get("d24", float("nan"))),
            "target_ell4": float(target_row.get("ell4", float("nan"))),
            "target_hrcg_anomaly": float(target_row.get("hrcg_anomaly", float("nan"))),
            "target_score": float(target_row.get("target_score", float("nan"))),
            "original_sentence": original_sentence,
            "candidate_sentence": candidate_sentence,
            "original_local": original_local,
            "candidate_local": candidate_local,
            "candidate_text": candidate_text,
            "mlm_ok": avg_probability >= float(min_mlm_probability),
            "local_semantic_ok": local_similarity >= float(min_local_similarity),
            "sentence_semantic_ok": sentence_similarity >= float(min_sentence_similarity),
            "pos_agreement_ok": pos_agreement >= float(min_pos_agreement),
            "local_h4_improved": local_h4_gain > float(min_local_h4_gain),
            "document_h4_ok": document_h4_change <= float(max_document_h4_increase),
            "edit_ratio_ok": edit_ratio <= float(max_edit_ratio),
            "semantic_validation_applied": False,
            "nli_ok": False,
            "cola_ok": False,
            "semantic_equivalence_ok": False,
            "valid_candidate": False,
            "selected_for_round": False,
        }
        preliminary = all([
            row["mlm_ok"], row["local_semantic_ok"], row["sentence_semantic_ok"],
            row["pos_agreement_ok"], row["numbers_preserved"], row["negation_preserved"],
            row["local_h4_improved"], row["document_h4_ok"], row["edit_ratio_ok"],
        ])
        row["preliminary_valid_candidate"] = bool(preliminary)
        rows.append(row)

    validation_indices = [i for i, row in enumerate(rows) if row["preliminary_valid_candidate"]]
    if validation_indices:
        validations = semantic_gate.score(
            [rows[i]["original_sentence"] for i in validation_indices],
            [rows[i]["candidate_sentence"] for i in validation_indices],
        )
        for idx, validation in zip(validation_indices, validations):
            rows[idx].update(validation)
            rows[idx]["semantic_validation_applied"] = True
            rows[idx]["valid_candidate"] = bool(validation["semantic_equivalence_ok"])

    for row in rows:
        if row["valid_candidate"]:
            row["rejection_reason"] = ""
            continue
        reasons: List[str] = []
        checks = [
            ("mlm_ok", "low_mlm_probability"),
            ("local_semantic_ok", "local_semantic_below_threshold"),
            ("sentence_semantic_ok", "sentence_semantic_below_threshold"),
            ("pos_agreement_ok", "span_pos_agreement_below_threshold"),
            ("numbers_preserved", "numbers_changed"),
            ("negation_preserved", "negation_changed"),
            ("local_h4_improved", "no_local_H4_improvement"),
            ("document_h4_ok", "document_H4_degraded"),
            ("edit_ratio_ok", "edit_ratio_budget_exceeded"),
        ]
        for key, message in checks:
            if not bool(row.get(key, False)):
                reasons.append(message)
        if row.get("semantic_validation_applied"):
            if not row.get("nli_ok"):
                reasons.append("bidirectional_NLI_failed")
            if not row.get("cola_ok"):
                reasons.append("grammatical_acceptability_failed")
        row["rejection_reason"] = ";".join(reasons)
    return rows


def _select_best_span_candidate(frame: pd.DataFrame) -> Optional[pd.Series]:
    valid = frame.loc[frame["valid_candidate"]].copy()
    if valid.empty:
        return None
    return valid.sort_values(
        ["local_H4_gain", "changed_token_count", "sentence_similarity", "mlm_log_probability"],
        ascending=[False, False, False, False],
    ).iloc[0]


def _load_span_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    print(f"[LOAD] UMBC 2-gram: {KENLM_2}")
    model2 = kenlm.Model(str(KENLM_2))
    print(f"[LOAD] UMBC 4-gram: {KENLM_4}")
    model4 = kenlm.Model(str(KENLM_4))
    mlm_path = resolve_mlm_path(args.mlm_model)
    device = choose_torch_device(args.device)
    tokenizer, masked_lm = load_masked_lm(mlm_path, device)
    print(f"[LOAD] Semantic model: {SEMANTIC_MODEL}")
    semantic_model = SentenceTransformer(
        str(SEMANTIC_MODEL), device="cpu", local_files_only=True
    )
    validation_device = choose_torch_device(args.validation_device)
    nli_path = resolve_validation_model_path(
        args.nli_model,
        env_name="GRAMGUARD_NLI_MODEL",
        repo_cache_name="models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli",
        display_name="NLI model",
    )
    cola_path = resolve_validation_model_path(
        args.cola_model,
        env_name="GRAMGUARD_COLA_MODEL",
        repo_cache_name="models--textattack--roberta-base-CoLA",
        display_name="CoLA model",
    )
    semantic_gate = SemanticEquivalenceGate(
        nli_path=nli_path,
        cola_path=cola_path,
        device=validation_device,
        min_entailment=args.min_bidirectional_entailment,
        max_contradiction=args.max_bidirectional_contradiction,
        min_cola_acceptability=args.min_cola_acceptability,
        max_cola_drop=args.max_cola_drop,
    )
    pos_tagger = OptionalPosTagger()
    if not pos_tagger.available:
        raise RuntimeError(f"Mandatory NLTK POS tagger unavailable: {pos_tagger.reason}")
    return {
        "model2": model2, "model4": model4,
        "tokenizer": tokenizer, "masked_lm": masked_lm, "device": device,
        "semantic_model": semantic_model, "semantic_gate": semantic_gate,
        "pos_tagger": pos_tagger,
    }


def run_span_rewrite_one(
    args: argparse.Namespace,
    *,
    raw_path: Path,
    texts: Sequence[str],
    sample_index: int,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    condition = safe_name(raw_path, args.source)
    stem = f"{condition}_sample{sample_index}"
    final_path = output_dir / f"lexical_final_{stem}.txt"
    summary_path = output_dir / f"lexical_summary_{stem}.csv"
    selected_path = output_dir / f"lexical_selected_{stem}.csv"
    targets_path = output_dir / f"lexical_targets_{stem}.csv"
    candidates_path = output_dir / f"lexical_candidates_{stem}.csv"

    if final_path.exists() and summary_path.exists() and final_path.stat().st_size > 0:
        print(f"[RESUME] sample={sample_index} already complete")
        return pd.read_csv(summary_path).iloc[0].to_dict()

    original_text = clean_text(texts[sample_index])
    working_text = original_text
    model2, model4 = runtime["model2"], runtime["model4"]
    initial_doc = score_document(model2, model4, original_text)
    all_targets: List[pd.DataFrame] = []
    all_candidates: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    changed_positions_total: set[int] = set()
    protected_positions: set[int] = set()
    stop_reason = "maximum span edit budget reached"

    print("\n" + "=" * 100)
    print("GramGuard-V3 | Detector-Independent HRCG-Guided Masked-Span Reconstruction")
    print("=" * 100)
    print(f"Selector    : {args.span_selector}")
    print(f"Sample      : {sample_index}")
    print(f"Words       : {word_count(original_text)}")
    print(f"Initial H4  : {initial_doc['H4']:.10f}")

    for round_index in range(1, int(args.max_span_edits) + 1):
        current_doc = score_document(model2, model4, working_text)
        attribution = full_token_attribution(model2, model4, working_text)
        verify_attribution(current_doc, attribution, working_text)
        rng = random.Random(int(args.seed) + sample_index * 1009 + round_index * 9173)
        targets, center, scale = select_span_targets(
            attribution, working_text, runtime["pos_tagger"], args.span_selector,
            args.span_target_count, args.min_word_length, protected_positions, rng,
        )
        if targets.empty:
            stop_reason = "no eligible span targets"
            break
        targets = targets.copy()
        targets["round"] = round_index
        all_targets.append(targets)
        print(f"\nROUND {round_index}: targets={len(targets)} selector={args.span_selector}")

        round_rows: List[Dict[str, Any]] = []
        words_now = clean_text(working_text).split()
        for target in targets.itertuples(index=False):
            center_position = int(target.position)
            raw_candidates, span_start, span_end, mask_positions = generate_masked_span_candidates(
                text=working_text,
                center_position=center_position,
                span_radius=args.span_radius,
                tokenizer=runtime["tokenizer"],
                masked_lm=runtime["masked_lm"],
                device=runtime["device"],
                pos_tagger=runtime["pos_tagger"],
                branch_k=args.span_branch_k,
                beam_width=args.span_beam_width,
                candidate_count=args.span_candidates,
                max_context_words=args.max_context_words,
            )
            print(
                f"  target={center_position}:{words_now[center_position]!r} "
                f"span={span_start}-{span_end} masks={mask_positions} candidates={len(raw_candidates)}"
            )
            if not raw_candidates:
                continue
            evaluated = evaluate_span_candidates(
                text=working_text,
                target_row=pd.Series(target._asdict()),
                raw_candidates=raw_candidates,
                span_start=span_start,
                span_end=span_end,
                model2=model2,
                model4=model4,
                semantic_model=runtime["semantic_model"],
                semantic_gate=runtime["semantic_gate"],
                pos_tagger=runtime["pos_tagger"],
                current_doc=current_doc,
                local_radius=args.local_radius,
                min_local_similarity=args.span_local_similarity,
                min_sentence_similarity=args.span_sentence_similarity,
                min_pos_agreement=args.span_min_pos_agreement,
                min_mlm_probability=args.span_min_mlm_probability,
                min_local_h4_gain=args.min_local_h4_gain,
                max_document_h4_increase=args.max_document_h4_increase,
                max_edit_ratio=args.max_edit_ratio,
                total_changed_positions=changed_positions_total,
            )
            for row in evaluated:
                row["round"] = round_index
                round_rows.append(row)
                all_candidates.append(row)

        if not round_rows:
            stop_reason = "masked-span generator produced no candidates"
            break
        round_df = pd.DataFrame(round_rows)
        best = _select_best_span_candidate(round_df)
        print(
            f"  evaluated={len(round_df)} valid={int(round_df['valid_candidate'].sum())}"
        )
        if best is None:
            stop_reason = "no span candidate passed semantic/H4/NLI/grammar safeguards"
            break

        selected_record = best.to_dict()
        selected_record["round"] = round_index
        selected_record["text_before"] = working_text
        selected_record["text_after"] = str(best["candidate_text"])
        selected_rows.append(selected_record)
        working_text = str(best["candidate_text"])
        new_changed = {int(x) for x in str(best["changed_positions"]).split("|") if x != ""}
        changed_positions_total |= new_changed
        # Prevent immediate overlapping edits around the same local region.
        for p in range(max(0, int(best["span_start"]) - 1), min(len(words_now), int(best["span_end"]) + 2)):
            protected_positions.add(p)
        print(
            f"  SELECT span {best['span_start']}-{best['span_end']}: "
            f"{best['original_span']!r} -> {best['candidate_span']!r}; "
            f"changed={best['changed_token_count']} H4gain={best['local_H4_gain']:.6f} "
            f"sent_sim={best['sentence_similarity']:.6f}"
        )

    final_doc = score_document(model2, model4, working_text)
    targets_df = pd.concat(all_targets, ignore_index=True) if all_targets else pd.DataFrame()
    candidates_df = pd.DataFrame(all_candidates)
    selected_df = pd.DataFrame(selected_rows)
    targets_df.to_csv(targets_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    final_path.write_text(working_text, encoding="utf-8")

    summary = {
        "mode": "span_rewrite",
        "span_selector": args.span_selector,
        "input_file": str(raw_path),
        "source": args.source,
        "sample_index": int(sample_index),
        "candidate_count": int(len(candidates_df)),
        "valid_candidate_count": int(candidates_df["valid_candidate"].sum()) if not candidates_df.empty else 0,
        "selected_edit_count": int(len(selected_df)),
        "applied_edit_count": int(len(selected_df)),
        "changed_token_count": int(len(changed_positions_total)),
        "actual_edit_ratio": float(len(changed_positions_total) / max(1, word_count(original_text))),
        "initial_H2": float(initial_doc["H2"]),
        "initial_H4": float(initial_doc["H4"]),
        "initial_A24": float(initial_doc["A24"]),
        "final_H2": float(final_doc["H2"]),
        "final_H4": float(final_doc["H4"]),
        "final_A24": float(final_doc["A24"]),
        "stop_reason": stop_reason,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(
        f"[DONE] sample={sample_index} span_edits={summary['applied_edit_count']} "
        f"changed_tokens={summary['changed_token_count']} ratio={summary['actual_edit_ratio']:.4f} "
        f"H4 {summary['initial_H4']:.6f}->{summary['final_H4']:.6f}"
    )
    return summary


def run_span_rewrite_batch(args: argparse.Namespace) -> None:
    if not args.allow_rewrite:
        raise RuntimeError("--mode span_rewrite requires --allow_rewrite")
    raw_path = Path(args.raw_file).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    texts = extract_texts(raw_path, args.source)
    indices = parse_index_spec(args.sample_indices)
    if not indices:
        indices = [int(args.sample_index)]
    invalid = [i for i in indices if i < 0 or i >= len(texts)]
    if invalid:
        raise IndexError(f"Invalid sample indices: {invalid}; available={len(texts)}")
    runtime = _load_span_runtime(args)
    summaries: List[Dict[str, Any]] = []
    for number, sample_index in enumerate(indices, start=1):
        print(f"\n{'#' * 100}\nSPAN BATCH {number}/{len(indices)} | sample={sample_index}\n{'#' * 100}")
        summaries.append(run_span_rewrite_one(
            args, raw_path=raw_path, texts=texts, sample_index=sample_index, runtime=runtime
        ))
    output_dir = Path(args.output_dir).expanduser()
    batch_path = output_dir / "batch_generation_summary.csv"
    pd.DataFrame(summaries).sort_values("sample_index").to_csv(batch_path, index=False)
    print(f"\nSaved batch generation summary: {batch_path}")




# -----------------------------------------------------------------------------
# FINAL IDEA-3: HRCG-guided human-null budget curve
# -----------------------------------------------------------------------------


def _ci_column(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    mapping = {str(c).lower(): c for c in frame.columns}
    for name in names:
        if str(name).lower() in mapping:
            return mapping[str(name).lower()]
    return None


def load_human_null_calibration(path: Path) -> pd.DataFrame:
    """Load the fixed human-only calibration table needed by the Idea-3 objective."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Human-null calibration CSV not found: {path}\n"
            "Set --human_null_csv to the fixed UMBC human calibration table."
        )
    raw = pd.read_csv(path)
    length_col = _ci_column(
        raw, ["length", "num_words", "word_count", "n_words", "tokens", "num_tokens"]
    )
    a24_col = _ci_column(
        raw, ["A24", "Source_A", "source_A", "A", "Global_HighRelG24", "rel_g24", "HighRelG24"]
    )
    h2_col = _ci_column(raw, ["H2", "h2", "H2_source", "source_H2"])
    h4_col = _ci_column(raw, ["H4", "h4", "H4_source", "source_H4"])
    text_col = _ci_column(raw, ["text", "source_text", "original", "sampled", "human"])

    if length_col is None and text_col is None:
        raise ValueError(
            "Human-null CSV must contain a length/word_count column or a human text column."
        )
    out = pd.DataFrame()
    if length_col is not None:
        out["length"] = pd.to_numeric(raw[length_col], errors="coerce")
    else:
        out["length"] = raw[text_col].fillna("").astype(str).map(word_count)

    if a24_col is not None:
        out["A24"] = pd.to_numeric(raw[a24_col], errors="coerce")
    elif h2_col is not None and h4_col is not None:
        h2 = pd.to_numeric(raw[h2_col], errors="coerce")
        h4 = pd.to_numeric(raw[h4_col], errors="coerce")
        out["A24"] = (h2 - h4) / h2.clip(lower=1e-12)
    else:
        raise ValueError(
            "Human-null CSV must contain A24 or both H2 and H4. "
            f"Available columns: {list(raw.columns)}"
        )

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["length", "A24"])
    out = out.loc[out["length"] >= 5].copy()
    out["length"] = out["length"].astype(int)
    if len(out) < 100:
        raise RuntimeError(f"Human-null calibration is unexpectedly small: {len(out)} rows")
    print(f"[LOAD] Human-null calibration: {path} rows={len(out):,}")
    return out.reset_index(drop=True)


def _human_null_robust_scale(values: np.ndarray) -> float:
    """Paper-2-compatible IQR/std fallback scale."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 1.0
    sd = float(np.nanstd(values, ddof=0))
    q75, q25 = np.nanpercentile(values, [75, 25])
    iqr = float((q75 - q25) / 1.349) if np.isfinite(q75 - q25) else float("nan")
    candidates = [x for x in (iqr, sd) if np.isfinite(x) and x > 1e-12]
    return max(candidates) if candidates else 1.0


def human_null_stats(
    *,
    length: int,
    a24: float,
    calibration: pd.DataFrame,
    rel_window: float,
    min_neighbors: int,
) -> Dict[str, float]:
    lengths = calibration["length"].to_numpy(dtype=float)
    values = calibration["A24"].to_numpy(dtype=float)
    L = float(length)
    lo = max(1.0, L * (1.0 - float(rel_window)))
    hi = L * (1.0 + float(rel_window))
    idx = np.where((lengths >= lo) & (lengths <= hi))[0]
    if len(idx) < int(min_neighbors):
        log_lengths = np.log1p(lengths)
        idx = np.argsort(np.abs(log_lengths - math.log1p(L)))[: int(min_neighbors)]
    local = values[idx]
    local = local[np.isfinite(local)]
    if local.size == 0:
        raise RuntimeError(f"No usable human-null neighbors for length={length}")
    center = float(np.nanmedian(local))
    scale = _human_null_robust_scale(local)
    signed_z = float((float(a24) - center) / (scale + 1e-12))
    return {
        "human_null_center": center,
        "human_null_scale": scale,
        "human_null_neighbors": int(local.size),
        "human_null_signed_z": signed_z,
        "human_null_distance": abs(signed_z),
    }


def evaluate_budget_curve_candidates(
    *,
    original_document_text: str,
    text: str,
    target_row: pd.Series,
    raw_candidates: List[Dict[str, Any]],
    span_start: int,
    span_end: int,
    model2: kenlm.Model,
    model4: kenlm.Model,
    semantic_model: SentenceTransformer,
    semantic_gate: SemanticEquivalenceGate,
    pos_tagger: OptionalPosTagger,
    current_doc: Dict[str, float],
    human_calibration: pd.DataFrame,
    human_rel_window: float,
    human_min_neighbors: int,
    local_radius: int,
    min_local_similarity: float,
    min_sentence_similarity: float,
    min_document_similarity: float,
    min_pos_agreement: float,
    min_mlm_probability: float,
    max_edit_ratio: float,
    total_changed_positions: set[int],
) -> List[Dict[str, Any]]:
    if not raw_candidates:
        return []

    words_before = clean_text(text).split()
    original_doc_clean = clean_text(original_document_text)
    center_position = int(target_row["position"])
    sent_start, sent_end = sentence_bounds(words_before, center_position)
    local_start = max(0, span_start - int(local_radius))
    local_end = min(len(words_before) - 1, span_end + int(local_radius))
    original_sentence = slice_words(words_before, sent_start, sent_end)
    original_local = slice_words(words_before, local_start, local_end)
    original_span = slice_words(words_before, span_start, span_end)
    original_local_h4 = kenlm_cross_entropy(model4, original_local)

    current_null = human_null_stats(
        length=word_count(text),
        a24=float(current_doc["A24"]),
        calibration=human_calibration,
        rel_window=human_rel_window,
        min_neighbors=human_min_neighbors,
    )

    candidate_sentences: List[str] = []
    candidate_locals: List[str] = []
    candidate_texts: List[str] = []
    for raw in raw_candidates:
        ctext = clean_text(raw["candidate_text"])
        cwords = ctext.split()
        candidate_texts.append(ctext)
        candidate_sentences.append(slice_words(cwords, sent_start, sent_end))
        candidate_locals.append(slice_words(cwords, local_start, local_end))

    local_sims = semantic_scores(
        semantic_model, [original_local] * len(raw_candidates), candidate_locals
    )
    sentence_sims = semantic_scores(
        semantic_model, [original_sentence] * len(raw_candidates), candidate_sentences
    )
    document_sims = semantic_scores(
        semantic_model, [original_doc_clean] * len(raw_candidates), candidate_texts
    )

    rows: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_candidates):
        candidate_text = candidate_texts[i]
        words_after = candidate_text.split()
        changed_positions = list(raw["changed_positions"])
        cumulative_changed = set(total_changed_positions) | set(changed_positions)
        edit_ratio = len(cumulative_changed) / max(1, word_count(original_document_text))
        candidate_sentence = candidate_sentences[i]
        candidate_local = candidate_locals[i]
        candidate_span = slice_words(words_after, span_start, span_end)
        local_h4_after = kenlm_cross_entropy(model4, candidate_local)
        local_h4_gain = float(original_local_h4 - local_h4_after)
        candidate_doc = score_document(model2, model4, candidate_text)
        candidate_null = human_null_stats(
            length=word_count(candidate_text),
            a24=float(candidate_doc["A24"]),
            calibration=human_calibration,
            rel_window=human_rel_window,
            min_neighbors=human_min_neighbors,
        )
        human_gain = float(
            current_null["human_null_distance"] - candidate_null["human_null_distance"]
        )
        pos_agreement = _span_pos_agreement(
            pos_tagger, words_before, words_after, changed_positions
        )
        avg_probability = float(
            math.exp(raw["mlm_log_probability"] / max(1, len(raw["replacement_map"])))
        )
        no_reedit = not bool(set(changed_positions) & set(total_changed_positions))

        row: Dict[str, Any] = {
            "target_rank": int(target_row["target_rank"]),
            "span_selector": "hrcg",
            "position": center_position,
            "span_start": int(span_start),
            "span_end": int(span_end),
            "original_span": original_span,
            "candidate_span": candidate_span,
            "changed_positions": "|".join(map(str, changed_positions)),
            "changed_token_count": int(len(changed_positions)),
            "cumulative_changed_token_count": int(len(cumulative_changed)),
            "actual_edit_ratio": float(edit_ratio),
            "mlm_log_probability": float(raw["mlm_log_probability"]),
            "mlm_geomean_probability": avg_probability,
            "local_context_similarity": float(local_sims[i]),
            "sentence_similarity": float(sentence_sims[i]),
            "document_similarity_to_original": float(document_sims[i]),
            "span_pos_agreement": float(pos_agreement),
            "numbers_preserved": number_signature(original_sentence) == number_signature(candidate_sentence),
            "negation_preserved": negation_signature(original_sentence) == negation_signature(candidate_sentence),
            "no_reedit": bool(no_reedit),
            "local_H4_before": float(original_local_h4),
            "local_H4_after": float(local_h4_after),
            "local_H4_gain_analysis_only": local_h4_gain,
            "document_H4_before": float(current_doc["H4"]),
            "document_H4_after": float(candidate_doc["H4"]),
            "document_H4_change_analysis_only": float(candidate_doc["H4"] - current_doc["H4"]),
            "A24_before": float(current_doc["A24"]),
            "A24_after": float(candidate_doc["A24"]),
            "human_null_center_before": float(current_null["human_null_center"]),
            "human_null_scale_before": float(current_null["human_null_scale"]),
            "human_null_signed_z_before": float(current_null["human_null_signed_z"]),
            "human_null_distance_before": float(current_null["human_null_distance"]),
            "human_null_center_after": float(candidate_null["human_null_center"]),
            "human_null_scale_after": float(candidate_null["human_null_scale"]),
            "human_null_signed_z_after": float(candidate_null["human_null_signed_z"]),
            "human_null_distance_after": float(candidate_null["human_null_distance"]),
            "human_null_distance_improvement": human_gain,
            "target_d24": float(target_row.get("d24", float("nan"))),
            "target_hrcg_anomaly": float(target_row.get("hrcg_anomaly", float("nan"))),
            "target_score": float(target_row.get("target_score", float("nan"))),
            "original_sentence": original_sentence,
            "candidate_sentence": candidate_sentence,
            "candidate_text": candidate_text,
            "mlm_ok": avg_probability >= float(min_mlm_probability),
            "local_semantic_ok": float(local_sims[i]) >= float(min_local_similarity),
            "sentence_semantic_ok": float(sentence_sims[i]) >= float(min_sentence_similarity),
            "document_semantic_ok": float(document_sims[i]) >= float(min_document_similarity),
            "pos_agreement_ok": float(pos_agreement) >= float(min_pos_agreement),
            "edit_ratio_ok": edit_ratio <= float(max_edit_ratio) + 1e-12,
            "human_alignment_improved": human_gain > 0.0,
            "semantic_validation_applied": False,
            "nli_ok": False,
            "cola_ok": False,
            "semantic_equivalence_ok": False,
            "valid_candidate": False,
            "selected_for_round": False,
        }
        preliminary = all([
            row["mlm_ok"], row["local_semantic_ok"], row["sentence_semantic_ok"],
            row["document_semantic_ok"], row["pos_agreement_ok"],
            row["numbers_preserved"], row["negation_preserved"], row["no_reedit"],
            row["edit_ratio_ok"], row["human_alignment_improved"],
        ])
        row["preliminary_valid_candidate"] = bool(preliminary)
        rows.append(row)

    validation_indices = [i for i, row in enumerate(rows) if row["preliminary_valid_candidate"]]
    if validation_indices:
        validations = semantic_gate.score(
            [rows[i]["original_sentence"] for i in validation_indices],
            [rows[i]["candidate_sentence"] for i in validation_indices],
        )
        for idx, validation in zip(validation_indices, validations):
            rows[idx].update(validation)
            rows[idx]["semantic_validation_applied"] = True
            rows[idx]["valid_candidate"] = bool(validation["semantic_equivalence_ok"])

    for row in rows:
        if row["valid_candidate"]:
            row["rejection_reason"] = ""
            continue
        reasons: List[str] = []
        checks = [
            ("mlm_ok", "low_mlm_probability"),
            ("local_semantic_ok", "local_semantic_below_threshold"),
            ("sentence_semantic_ok", "sentence_semantic_below_threshold"),
            ("document_semantic_ok", "document_semantic_below_threshold"),
            ("pos_agreement_ok", "span_pos_agreement_below_threshold"),
            ("numbers_preserved", "numbers_changed"),
            ("negation_preserved", "negation_changed"),
            ("no_reedit", "previously_changed_position_reedited"),
            ("edit_ratio_ok", "edit_ratio_budget_exceeded"),
            ("human_alignment_improved", "no_human_null_alignment_improvement"),
        ]
        for key, message in checks:
            if not bool(row.get(key, False)):
                reasons.append(message)
        if row.get("semantic_validation_applied"):
            if not row.get("nli_ok"):
                reasons.append("bidirectional_NLI_failed")
            if not row.get("cola_ok"):
                reasons.append("grammatical_acceptability_failed")
        row["rejection_reason"] = ";".join(reasons)
    return rows


def _select_best_budget_candidate(frame: pd.DataFrame) -> Optional[pd.Series]:
    valid = frame.loc[frame["valid_candidate"]].copy()
    if valid.empty:
        return None
    return valid.sort_values(
        [
            "human_null_distance_improvement",
            "document_similarity_to_original",
            "sentence_similarity",
            "changed_token_count",
            "mlm_log_probability",
        ],
        ascending=[False, False, False, True, False],
    ).iloc[0]


def _budget_label(value: float) -> str:
    return f"{int(round(float(value) * 100)):02d}pct"


def _write_budget_snapshot(
    *,
    root: Path,
    budget: float,
    stem: str,
    text: str,
    original_text: str,
    initial_doc: Dict[str, float],
    final_doc: Dict[str, float],
    initial_null: Dict[str, float],
    final_null: Dict[str, float],
    selected_rows: List[Dict[str, Any]],
    changed_positions: set[int],
    stage_stop_reason: str,
    raw_path: Path,
    source: str,
    sample_index: int,
) -> Dict[str, Any]:
    label = _budget_label(budget)
    out_dir = root / f"budget_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"lexical_final_{stem}.txt"
    summary_path = out_dir / f"lexical_summary_{stem}.csv"
    selected_path = out_dir / f"lexical_selected_{stem}.csv"
    final_path.write_text(text, encoding="utf-8")
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(selected_path, index=False)
    summary = {
        "mode": "budget_curve",
        "span_selector": "hrcg",
        "budget": float(budget),
        "budget_percent": int(round(float(budget) * 100)),
        "input_file": str(raw_path),
        "source": source,
        "sample_index": int(sample_index),
        "word_count": int(word_count(original_text)),
        "selected_edit_count": int(len(selected_rows)),
        "applied_edit_count": int(len(selected_rows)),
        "changed_token_count": int(len(changed_positions)),
        "actual_edit_ratio": float(len(changed_positions) / max(1, word_count(original_text))),
        "initial_H2": float(initial_doc["H2"]),
        "initial_H4": float(initial_doc["H4"]),
        "initial_A24": float(initial_doc["A24"]),
        "final_H2": float(final_doc["H2"]),
        "final_H4": float(final_doc["H4"]),
        "final_A24": float(final_doc["A24"]),
        "initial_human_null_signed_z": float(initial_null["human_null_signed_z"]),
        "initial_human_null_distance": float(initial_null["human_null_distance"]),
        "final_human_null_signed_z": float(final_null["human_null_signed_z"]),
        "final_human_null_distance": float(final_null["human_null_distance"]),
        "human_null_distance_improvement": float(
            initial_null["human_null_distance"] - final_null["human_null_distance"]
        ),
        "stop_reason": stage_stop_reason,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary


def run_budget_curve_one(
    args: argparse.Namespace,
    *,
    raw_path: Path,
    texts: Sequence[str],
    sample_index: int,
    runtime: Dict[str, Any],
    human_calibration: pd.DataFrame,
    budgets: Sequence[float],
) -> List[Dict[str, Any]]:
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    condition = safe_name(raw_path, args.source)
    stem = f"{condition}_sample{sample_index}"
    trajectory_summary_path = root / f"budget_curve_trajectory_summary_{stem}.csv"
    trajectory_selected_path = root / f"budget_curve_selected_{stem}.csv"
    trajectory_candidates_path = root / f"budget_curve_candidates_{stem}.csv"
    trajectory_targets_path = root / f"budget_curve_targets_{stem}.csv"

    # Resume only when every requested budget snapshot exists.
    all_complete = True
    resumed: List[Dict[str, Any]] = []
    for budget in budgets:
        bdir = root / f"budget_{_budget_label(budget)}"
        spath = bdir / f"lexical_summary_{stem}.csv"
        fpath = bdir / f"lexical_final_{stem}.txt"
        if not (spath.exists() and fpath.exists() and fpath.stat().st_size > 0):
            all_complete = False
            break
        resumed.append(pd.read_csv(spath).iloc[0].to_dict())
    if all_complete and len(resumed) == len(budgets):
        print(f"[RESUME] sample={sample_index} all budget snapshots already complete")
        return resumed

    original_text = clean_text(texts[sample_index])
    working_text = original_text
    model2, model4 = runtime["model2"], runtime["model4"]
    initial_doc = score_document(model2, model4, original_text)
    initial_null = human_null_stats(
        length=word_count(original_text),
        a24=float(initial_doc["A24"]),
        calibration=human_calibration,
        rel_window=args.human_null_rel_window,
        min_neighbors=args.human_null_min_neighbors,
    )
    changed_positions_total: set[int] = set()
    protected_positions: set[int] = set()
    all_targets: List[pd.DataFrame] = []
    all_candidates: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    snapshot_summaries: List[Dict[str, Any]] = []

    print("\n" + "=" * 110)
    print("GramGuard-V3 FINAL | HRCG-Guided Human-Null Budget Curve")
    print("=" * 110)
    print(f"Sample                : {sample_index}")
    print(f"Words                 : {word_count(original_text)}")
    print(f"Initial A24           : {initial_doc['A24']:.10f}")
    print(f"Human-null center     : {initial_null['human_null_center']:.10f}")
    print(f"Initial signed z      : {initial_null['human_null_signed_z']:.6f}")
    print(f"Initial |z| distance  : {initial_null['human_null_distance']:.6f}")
    print(f"Budgets               : {[int(b*100) for b in budgets]}%")

    global_round = 0
    for budget in budgets:
        stage_round = 0
        stage_stop = "budget ceiling reached"
        while stage_round < int(args.max_rounds_per_budget):
            current_ratio = len(changed_positions_total) / max(1, word_count(original_text))
            if current_ratio >= float(budget) - 1e-12:
                stage_stop = "budget ceiling reached"
                break

            current_doc = score_document(model2, model4, working_text)
            attribution = full_token_attribution(model2, model4, working_text)
            verify_attribution(current_doc, attribution, working_text)
            rng = random.Random(int(args.seed) + sample_index * 1009 + (global_round + 1) * 9173)
            targets, _, _ = select_span_targets(
                attribution,
                working_text,
                runtime["pos_tagger"],
                "hrcg",
                args.span_target_count,
                args.min_word_length,
                protected_positions,
                rng,
            )
            if targets.empty:
                stage_stop = "no eligible HRCG span targets"
                break
            global_round += 1
            stage_round += 1
            targets = targets.copy()
            targets["global_round"] = global_round
            targets["budget"] = float(budget)
            all_targets.append(targets)

            round_rows: List[Dict[str, Any]] = []
            words_now = clean_text(working_text).split()
            for target in targets.itertuples(index=False):
                center_position = int(target.position)
                raw_candidates, span_start, span_end, mask_positions = generate_masked_span_candidates(
                    text=working_text,
                    center_position=center_position,
                    span_radius=args.span_radius,
                    tokenizer=runtime["tokenizer"],
                    masked_lm=runtime["masked_lm"],
                    device=runtime["device"],
                    pos_tagger=runtime["pos_tagger"],
                    branch_k=args.span_branch_k,
                    beam_width=args.span_beam_width,
                    candidate_count=args.span_candidates,
                    max_context_words=args.max_context_words,
                )
                if not raw_candidates:
                    continue
                evaluated = evaluate_budget_curve_candidates(
                    original_document_text=original_text,
                    text=working_text,
                    target_row=pd.Series(target._asdict()),
                    raw_candidates=raw_candidates,
                    span_start=span_start,
                    span_end=span_end,
                    model2=model2,
                    model4=model4,
                    semantic_model=runtime["semantic_model"],
                    semantic_gate=runtime["semantic_gate"],
                    pos_tagger=runtime["pos_tagger"],
                    current_doc=current_doc,
                    human_calibration=human_calibration,
                    human_rel_window=args.human_null_rel_window,
                    human_min_neighbors=args.human_null_min_neighbors,
                    local_radius=args.local_radius,
                    min_local_similarity=args.span_local_similarity,
                    min_sentence_similarity=args.span_sentence_similarity,
                    min_document_similarity=args.min_document_similarity,
                    min_pos_agreement=args.span_min_pos_agreement,
                    min_mlm_probability=args.span_min_mlm_probability,
                    max_edit_ratio=float(budget),
                    total_changed_positions=changed_positions_total,
                )
                for row in evaluated:
                    row["global_round"] = global_round
                    row["budget"] = float(budget)
                    round_rows.append(row)
                    all_candidates.append(row)

            if not round_rows:
                stage_stop = "span reconstructor produced no candidates"
                break
            round_df = pd.DataFrame(round_rows)
            best = _select_best_budget_candidate(round_df)
            valid_count = int(round_df["valid_candidate"].sum())
            print(
                f"  budget={int(budget*100):02d}% round={stage_round:02d} "
                f"ratio={current_ratio:.4f} evaluated={len(round_df)} valid={valid_count}"
            )
            if best is None:
                stage_stop = "no human-null-improving candidate passed quality gates within budget"
                break

            best_record = best.to_dict()
            # Compatibility with the existing paired evaluator's compact edit description.
            best_record["original_core"] = str(best_record.get("original_span", ""))
            best_record["candidate_core"] = str(best_record.get("candidate_span", ""))
            best_record["global_round"] = global_round
            best_record["budget_stage"] = float(budget)
            best_record["text_before"] = working_text
            best_record["text_after"] = str(best["candidate_text"])
            best_record["selected_for_round"] = True
            selected_rows.append(best_record)
            working_text = str(best["candidate_text"])
            new_changed = {int(x) for x in str(best["changed_positions"]).split("|") if x != ""}
            changed_positions_total |= new_changed
            # Only the already changed positions are protected. Nearby HRCG regions remain editable.
            protected_positions |= new_changed
            print(
                f"    SELECT {best['original_span']!r} -> {best['candidate_span']!r}; "
                f"changed={len(new_changed)} cumulative={len(changed_positions_total)} "
                f"ratio={len(changed_positions_total)/max(1,word_count(original_text)):.4f} "
                f"|z| {best['human_null_distance_before']:.4f}->{best['human_null_distance_after']:.4f} "
                f"gain={best['human_null_distance_improvement']:.4f}"
            )

        final_doc = score_document(model2, model4, working_text)
        final_null = human_null_stats(
            length=word_count(working_text),
            a24=float(final_doc["A24"]),
            calibration=human_calibration,
            rel_window=args.human_null_rel_window,
            min_neighbors=args.human_null_min_neighbors,
        )
        snap = _write_budget_snapshot(
            root=root,
            budget=float(budget),
            stem=stem,
            text=working_text,
            original_text=original_text,
            initial_doc=initial_doc,
            final_doc=final_doc,
            initial_null=initial_null,
            final_null=final_null,
            selected_rows=selected_rows,
            changed_positions=changed_positions_total,
            stage_stop_reason=stage_stop,
            raw_path=raw_path,
            source=args.source,
            sample_index=sample_index,
        )
        snapshot_summaries.append(snap)
        print(
            f"[SNAPSHOT {int(budget*100):02d}%] edits={len(selected_rows)} "
            f"changed={len(changed_positions_total)} ratio={snap['actual_edit_ratio']:.4f} "
            f"human |z|={snap['final_human_null_distance']:.4f} stop={stage_stop}"
        )

    pd.concat(all_targets, ignore_index=True).to_csv(
        trajectory_targets_path, index=False
    ) if all_targets else pd.DataFrame().to_csv(trajectory_targets_path, index=False)
    pd.DataFrame(all_candidates).to_csv(trajectory_candidates_path, index=False)
    pd.DataFrame(selected_rows).to_csv(trajectory_selected_path, index=False)
    pd.DataFrame(snapshot_summaries).to_csv(trajectory_summary_path, index=False)
    return snapshot_summaries


def run_budget_curve_batch(args: argparse.Namespace) -> None:
    if not args.allow_rewrite:
        raise RuntimeError("--mode budget_curve requires --allow_rewrite")
    raw_path = Path(args.raw_file).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    texts = extract_texts(raw_path, args.source)
    indices = parse_index_spec(args.sample_indices)
    if not indices:
        indices = [int(args.sample_index)]
    invalid = [i for i in indices if i < 0 or i >= len(texts)]
    if invalid:
        raise IndexError(f"Invalid sample indices: {invalid}; available={len(texts)}")

    budgets = sorted({float(x.strip()) for x in str(args.edit_budgets).split(",") if x.strip()})
    if not budgets or any(b <= 0 or b > 0.60 for b in budgets):
        raise ValueError("--edit_budgets must contain fractions in (0, 0.60], e.g. 0.05,0.10,0.15,0.20")
    expected = [0.05, 0.10, 0.15, 0.20]
    if args.require_final_budgets and (
        len(budgets) != len(expected)
        or any(abs(a - b) > 1e-9 for a, b in zip(budgets, expected))
    ):
        raise ValueError("Final experiment is locked to --edit_budgets 0.05,0.10,0.15,0.20")

    runtime = _load_span_runtime(args)
    human_calibration = load_human_null_calibration(Path(args.human_null_csv))
    all_by_budget: Dict[float, List[Dict[str, Any]]] = {b: [] for b in budgets}
    for number, sample_index in enumerate(indices, start=1):
        print(f"\n{'#' * 110}\nFINAL BUDGET CURVE {number}/{len(indices)} | sample={sample_index}\n{'#' * 110}")
        snapshots = run_budget_curve_one(
            args,
            raw_path=raw_path,
            texts=texts,
            sample_index=sample_index,
            runtime=runtime,
            human_calibration=human_calibration,
            budgets=budgets,
        )
        for rec in snapshots:
            all_by_budget[float(rec["budget"])].append(rec)

    root = Path(args.output_dir).expanduser().resolve()
    master_rows: List[Dict[str, Any]] = []
    for budget in budgets:
        bdir = root / f"budget_{_budget_label(budget)}"
        frame = pd.DataFrame(all_by_budget[budget]).sort_values("sample_index")
        frame.to_csv(bdir / "batch_generation_summary.csv", index=False)
        achieved = pd.to_numeric(frame["actual_edit_ratio"], errors="coerce").fillna(0.0)
        edited = pd.to_numeric(frame["applied_edit_count"], errors="coerce").fillna(0)
        master_rows.append({
            "budget": budget,
            "budget_percent": int(round(budget*100)),
            "documents": int(len(frame)),
            "edited_documents": int((edited > 0).sum()),
            "coverage": float((edited > 0).mean()) if len(frame) else float("nan"),
            "total_span_edits": int(edited.sum()),
            "total_changed_tokens": int(pd.to_numeric(frame["changed_token_count"], errors="coerce").fillna(0).sum()),
            "mean_actual_edit_ratio": float(achieved.mean()) if len(frame) else float("nan"),
            "median_actual_edit_ratio": float(achieved.median()) if len(frame) else float("nan"),
            "mean_human_null_distance_before": float(pd.to_numeric(frame["initial_human_null_distance"], errors="coerce").mean()),
            "mean_human_null_distance_after": float(pd.to_numeric(frame["final_human_null_distance"], errors="coerce").mean()),
            "mean_human_null_distance_improvement": float(pd.to_numeric(frame["human_null_distance_improvement"], errors="coerce").mean()),
            "mean_A24_change": float((pd.to_numeric(frame["final_A24"]) - pd.to_numeric(frame["initial_A24"])).mean()),
            "mean_H4_change": float((pd.to_numeric(frame["final_H4"]) - pd.to_numeric(frame["initial_H4"])).mean()),
        })
    master = pd.DataFrame(master_rows)
    master_path = root / "budget_curve_generation_master.csv"
    master.to_csv(master_path, index=False)
    print("\n" + "="*110)
    print("FINAL BUDGET-CURVE GENERATION SUMMARY")
    print("="*110)
    print(master.to_string(index=False))
    print(f"\nSaved: {master_path}")




# -----------------------------------------------------------------------------
# FINAL LLM rewrite engine: HRCG localization -> instruct-LLM local rewrite
# -> human-null selection. This is the last development experiment.
# -----------------------------------------------------------------------------


def _repo_cache_name(repo_id: str) -> str:
    return "models--" + str(repo_id).replace("/", "--")


def _local_snapshot_for_repo(repo_id: str) -> Optional[Path]:
    root = Path.home() / ".cache" / "huggingface" / "hub" / _repo_cache_name(repo_id) / "snapshots"
    if not root.exists():
        return None
    usable = []
    for path in root.iterdir():
        if not path.is_dir() or not (path / "config.json").exists():
            continue
        if not _has_causal_weights(path) or not _has_causal_tokenizer(path):
            continue
        usable.append(path)
    return max(usable, key=lambda p: p.stat().st_mtime) if usable else None


def discover_local_instruct_models() -> List[Tuple[str, Path]]:
    # Prefer a small instruct model for the Mac development run. Larger models
    # remain supported explicitly through --rewrite_model.
    preferred = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    found: List[Tuple[str, Path]] = []
    for repo in preferred:
        snap = _local_snapshot_for_repo(repo)
        if snap is not None:
            found.append((repo, snap))
    return found


def resolve_rewrite_model(spec: str, allow_download: bool) -> Tuple[str, bool]:
    requested = str(spec).strip()
    if requested and requested.lower() != "auto":
        explicit = Path(os.path.expanduser(requested))
        if explicit.exists():
            return str(explicit.resolve()), True
        local = _local_snapshot_for_repo(requested)
        if local is not None:
            return str(local), True
        if allow_download and "/" in requested:
            return requested, False
        raise FileNotFoundError(
            f"Rewrite model not found locally: {requested}. "
            "Pass --allow_model_download or an absolute local model path."
        )

    env_model = os.environ.get("GRAMGUARD_REWRITE_MODEL", "").strip()
    if env_model:
        return resolve_rewrite_model(env_model, allow_download)

    local_models = discover_local_instruct_models()
    if local_models:
        repo, path = local_models[0]
        print(f"[AUTO] Rewrite model: {repo} -> {path}")
        return str(path), True

    default_repo = "Qwen/Qwen2.5-1.5B-Instruct"
    if allow_download:
        print(f"[AUTO] No local instruct model found; will download/cache {default_repo}")
        return default_repo, False
    raise FileNotFoundError(
        "No supported local instruct model was found. Use --rewrite_model /absolute/path, "
        "or add --allow_model_download to fetch Qwen/Qwen2.5-1.5B-Instruct."
    )


def load_rewrite_model(model_spec: str, *, local_only: bool, device: torch.device):
    print(f"[LOAD] Rewrite tokenizer: {model_spec}")
    tokenizer = AutoTokenizer.from_pretrained(model_spec, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32
    kwargs = {"local_files_only": local_only, "low_cpu_mem_usage": True}
    print(f"[LOAD] Rewrite model    : {model_spec}")
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_spec, dtype=dtype, **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_spec, torch_dtype=dtype, **kwargs)
    except ImportError:
        # Older/minimal Transformers installs may not have Accelerate available
        # for low_cpu_mem_usage. Retry without that optional optimization.
        kwargs.pop("low_cpu_mem_usage", None)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_spec, dtype=dtype, **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_spec, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    print(f"[LOAD] Rewrite device   : {device}")
    return tokenizer, model


def _normalized_budget_tokens(text: str) -> List[str]:
    tokens = []
    for token in clean_text(text).split():
        parts = split_token_parts(token)
        core = parts.core.lower().strip()
        tokens.append(core if core else token.lower())
    return tokens


def word_diff_count(original_text: str, candidate_text: str) -> int:
    """Word-level edit count robust to replacements, insertions and deletions."""
    a = _normalized_budget_tokens(original_text)
    b = _normalized_budget_tokens(candidate_text)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return int(changed)


def _capitalized_content_signature(sentence: str) -> Tuple[str, ...]:
    words = clean_text(sentence).split()
    protected = []
    for i, raw in enumerate(words):
        core = split_token_parts(raw).core
        if not core:
            continue
        # Ignore ordinary sentence-initial capitalization. Preserve internal
        # proper-name-like tokens and acronyms exactly.
        if (i > 0 and core[:1].isupper()) or (len(core) > 1 and core.isupper()):
            protected.append(core)
    return tuple(sorted(protected))


def _clean_llm_rewrite(raw: str) -> str:
    text = str(raw).strip()
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.I | re.S)
    if text.lstrip().lower().startswith("<think>"):
        return ""
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    prefixes = [
        "rewritten sentence:", "rewrite:", "rewritten:", "paraphrase:",
        "output:", "answer:", "target sentence:",
    ]
    lower = text.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # One target sentence only. Models sometimes add explanation on a new line.
    text = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return clean_text(text)


def _make_local_rewrite_prompt(
    *,
    previous_sentence: str,
    target_sentence: str,
    next_sentence: str,
    max_changed_words: int,
    desired_changed_words: int,
) -> List[Dict[str, str]]:
    system = (
        "You are a conservative local paraphraser. Rewrite ONLY the TARGET sentence. "
        "Preserve its factual meaning, entities, numbers, polarity/negation, tense, and logical relations. "
        "Do not summarize, add facts, delete facts, or explain your answer. Return only one rewritten sentence."
    )
    user = (
        f"PREVIOUS CONTEXT: {previous_sentence or '[none]'}\n"
        f"TARGET: {target_sentence}\n"
        f"NEXT CONTEXT: {next_sentence or '[none]'}\n\n"
        f"Rewrite the TARGET using meaning-preserving changes in wording and local syntax. "
        f"Aim to change about {desired_changed_words} words and do not intentionally change more than "
        f"{max_changed_words} words. Keep the sentence natural. Output only the rewritten TARGET sentence."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_llm_sentence_candidates(
    *,
    text: str,
    center_position: int,
    tokenizer,
    model,
    device: torch.device,
    candidate_count: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_changed_words: int,
    desired_changed_words: int,
) -> Tuple[List[Dict[str, Any]], int, int, str]:
    words = clean_text(text).split()
    sent_start, sent_end = sentence_bounds(words, center_position)
    target_sentence = slice_words(words, sent_start, sent_end)
    if not target_sentence:
        return [], sent_start, sent_end, target_sentence

    prev_sentence = ""
    if sent_start > 0:
        ps, pe = sentence_bounds(words, sent_start - 1)
        prev_sentence = slice_words(words, ps, pe)
    next_sentence = ""
    if sent_end + 1 < len(words):
        ns, ne = sentence_bounds(words, sent_end + 1)
        next_sentence = slice_words(words, ns, ne)

    messages = _make_local_rewrite_prompt(
        previous_sentence=prev_sentence,
        target_sentence=target_sentence,
        next_sentence=next_sentence,
        max_changed_words=max_changed_words,
        desired_changed_words=desired_changed_words,
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt = (
            messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n\nREWRITE:"
        )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_len = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=50,
            num_return_sequences=int(candidate_count),
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    seen = set()
    results: List[Dict[str, Any]] = []
    for rank, sequence in enumerate(generated, start=1):
        decoded = tokenizer.decode(sequence[input_len:], skip_special_tokens=True)
        rewritten_sentence = _clean_llm_rewrite(decoded)
        if not rewritten_sentence:
            continue
        if rewritten_sentence.lower() == target_sentence.lower():
            continue
        key = rewritten_sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        rewritten_words = rewritten_sentence.split()
        # Avoid obvious run-away generations before expensive validation.
        target_len = max(1, len(target_sentence.split()))
        if len(rewritten_words) < max(3, int(0.55 * target_len)):
            continue
        if len(rewritten_words) > int(1.65 * target_len) + 3:
            continue
        candidate_words = words[:sent_start] + rewritten_words + words[sent_end + 1:]
        candidate_text = clean_text(" ".join(candidate_words))
        results.append({
            "candidate_text": candidate_text,
            "candidate_sentence": rewritten_sentence,
            "generation_rank": int(rank),
            "candidate_source": "instruct_llm_local_sentence",
        })
    return results, sent_start, sent_end, target_sentence


def evaluate_llm_budget_candidates(
    *,
    original_document_text: str,
    current_text: str,
    target_row: pd.Series,
    raw_candidates: List[Dict[str, Any]],
    sentence_start: int,
    sentence_end: int,
    original_sentence: str,
    model2: kenlm.Model,
    model4: kenlm.Model,
    semantic_model: SentenceTransformer,
    semantic_gate: SemanticEquivalenceGate,
    current_doc: Dict[str, float],
    human_calibration: pd.DataFrame,
    human_rel_window: float,
    human_min_neighbors: int,
    min_sentence_similarity: float,
    min_document_similarity: float,
    max_edit_ratio: float,
    min_human_gain: float,
) -> List[Dict[str, Any]]:
    if not raw_candidates:
        return []
    current_words = clean_text(current_text).split()
    # Context-window similarity catches rewrites that preserve the sentence in
    # isolation but become incoherent with neighboring text.
    context_start = max(0, sentence_start - 20)
    context_end = min(len(current_words) - 1, sentence_end + 20)
    original_context = slice_words(current_words, context_start, context_end)
    original_doc_clean = clean_text(original_document_text)
    current_null = human_null_stats(
        length=word_count(current_text),
        a24=float(current_doc["A24"]),
        calibration=human_calibration,
        rel_window=human_rel_window,
        min_neighbors=human_min_neighbors,
    )

    candidate_sentences = [str(r["candidate_sentence"]) for r in raw_candidates]
    candidate_texts = [clean_text(r["candidate_text"]) for r in raw_candidates]
    contexts = []
    for ctext in candidate_texts:
        cwords = ctext.split()
        # Approximate corresponding context using the same left anchor and a
        # window large enough to include the rewritten sentence.
        ce = min(len(cwords) - 1, context_start + max(1, context_end - context_start) + 12)
        contexts.append(slice_words(cwords, context_start, ce))

    sentence_sims = semantic_scores(
        semantic_model, [original_sentence] * len(raw_candidates), candidate_sentences
    )
    context_sims = semantic_scores(
        semantic_model, [original_context] * len(raw_candidates), contexts
    )
    document_sims = semantic_scores(
        semantic_model, [original_doc_clean] * len(raw_candidates), candidate_texts
    )

    rows: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_candidates):
        candidate_text = candidate_texts[i]
        candidate_sentence = candidate_sentences[i]
        candidate_doc = score_document(model2, model4, candidate_text)
        candidate_null = human_null_stats(
            length=word_count(candidate_text),
            a24=float(candidate_doc["A24"]),
            calibration=human_calibration,
            rel_window=human_rel_window,
            min_neighbors=human_min_neighbors,
        )
        human_gain = float(current_null["human_null_distance"] - candidate_null["human_null_distance"])
        changed_count = word_diff_count(original_document_text, candidate_text)
        actual_ratio = changed_count / max(1, word_count(original_document_text))
        sentence_changed = word_diff_count(original_sentence, candidate_sentence)
        row: Dict[str, Any] = {
            "target_rank": int(target_row["target_rank"]),
            "span_selector": "hrcg",
            "position": int(target_row["position"]),
            "sentence_start": int(sentence_start),
            "sentence_end": int(sentence_end),
            "original_span": original_sentence,
            "candidate_span": candidate_sentence,
            "original_sentence": original_sentence,
            "candidate_sentence": candidate_sentence,
            "candidate_text": candidate_text,
            "candidate_source": str(raw.get("candidate_source", "instruct_llm_local_sentence")),
            "generation_rank": int(raw.get("generation_rank", i + 1)),
            "sentence_changed_word_count": int(sentence_changed),
            "cumulative_changed_token_count": int(changed_count),
            "actual_edit_ratio": float(actual_ratio),
            "context_similarity": float(context_sims[i]),
            "sentence_similarity": float(sentence_sims[i]),
            "document_similarity_to_original": float(document_sims[i]),
            "numbers_preserved": number_signature(original_sentence) == number_signature(candidate_sentence),
            "negation_preserved": negation_signature(original_sentence) == negation_signature(candidate_sentence),
            "proper_tokens_preserved": _capitalized_content_signature(original_sentence) == _capitalized_content_signature(candidate_sentence),
            "A24_before": float(current_doc["A24"]),
            "A24_after": float(candidate_doc["A24"]),
            "H4_before_analysis_only": float(current_doc["H4"]),
            "H4_after_analysis_only": float(candidate_doc["H4"]),
            "human_null_distance_before": float(current_null["human_null_distance"]),
            "human_null_distance_after": float(candidate_null["human_null_distance"]),
            "human_null_signed_z_before": float(current_null["human_null_signed_z"]),
            "human_null_signed_z_after": float(candidate_null["human_null_signed_z"]),
            "human_null_distance_improvement": float(human_gain),
            "target_d24": float(target_row.get("d24", float("nan"))),
            "target_hrcg_anomaly": float(target_row.get("hrcg_anomaly", float("nan"))),
            "target_score": float(target_row.get("target_score", float("nan"))),
            "sentence_semantic_ok": float(sentence_sims[i]) >= float(min_sentence_similarity),
            "context_semantic_ok": float(context_sims[i]) >= max(0.80, float(min_sentence_similarity) - 0.03),
            "document_semantic_ok": float(document_sims[i]) >= float(min_document_similarity),
            "edit_ratio_ok": float(actual_ratio) <= float(max_edit_ratio) + 1e-12,
            "human_alignment_improved": float(human_gain) > float(min_human_gain),
            "semantic_validation_applied": False,
            "nli_ok": False,
            "cola_ok": False,
            "semantic_equivalence_ok": False,
            "valid_candidate": False,
            "selected_for_round": False,
        }
        preliminary = all([
            row["sentence_semantic_ok"], row["context_semantic_ok"],
            row["document_semantic_ok"], row["numbers_preserved"],
            row["negation_preserved"], row["proper_tokens_preserved"],
            row["edit_ratio_ok"], row["human_alignment_improved"],
        ])
        row["preliminary_valid_candidate"] = bool(preliminary)
        rows.append(row)

    indices = [i for i, r in enumerate(rows) if r["preliminary_valid_candidate"]]
    if indices:
        validations = semantic_gate.score(
            [rows[i]["original_sentence"] for i in indices],
            [rows[i]["candidate_sentence"] for i in indices],
        )
        for idx, validation in zip(indices, validations):
            rows[idx].update(validation)
            rows[idx]["semantic_validation_applied"] = True
            rows[idx]["valid_candidate"] = bool(validation["semantic_equivalence_ok"])

    for row in rows:
        if row["valid_candidate"]:
            row["rejection_reason"] = ""
            continue
        reasons = []
        for key, label in [
            ("sentence_semantic_ok", "sentence_semantic_below_threshold"),
            ("context_semantic_ok", "context_semantic_below_threshold"),
            ("document_semantic_ok", "document_semantic_below_threshold"),
            ("numbers_preserved", "numbers_changed"),
            ("negation_preserved", "negation_changed"),
            ("proper_tokens_preserved", "proper_tokens_changed"),
            ("edit_ratio_ok", "edit_ratio_budget_exceeded"),
            ("human_alignment_improved", "no_human_null_alignment_improvement"),
        ]:
            if not bool(row.get(key, False)):
                reasons.append(label)
        if row.get("semantic_validation_applied"):
            if not row.get("nli_ok"):
                reasons.append("bidirectional_NLI_failed")
            if not row.get("cola_ok"):
                reasons.append("grammatical_acceptability_failed")
        row["rejection_reason"] = ";".join(reasons)
    return rows


def _select_best_llm_candidate(frame: pd.DataFrame) -> Optional[pd.Series]:
    valid = frame.loc[frame["valid_candidate"]].copy()
    if valid.empty:
        return None
    return valid.sort_values(
        [
            "human_null_distance_improvement",
            "document_similarity_to_original",
            "nli_min_bidirectional_entailment",
            "sentence_similarity",
            "sentence_changed_word_count",
        ],
        ascending=[False, False, False, False, False],
    ).iloc[0]


def _load_llm_budget_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    print(f"[LOAD] UMBC 2-gram: {KENLM_2}")
    model2 = kenlm.Model(str(KENLM_2))
    print(f"[LOAD] UMBC 4-gram: {KENLM_4}")
    model4 = kenlm.Model(str(KENLM_4))
    rewrite_device = choose_torch_device(args.rewrite_device)
    rewrite_spec, local_only = resolve_rewrite_model(args.rewrite_model, args.allow_model_download)
    rewrite_tokenizer, rewrite_model = load_rewrite_model(
        rewrite_spec, local_only=local_only, device=rewrite_device
    )
    print(f"[LOAD] Semantic model: {SEMANTIC_MODEL}")
    semantic_model = SentenceTransformer(str(SEMANTIC_MODEL), device="cpu", local_files_only=True)
    validation_device = choose_torch_device(args.validation_device)
    nli_path = resolve_validation_model_path(
        args.nli_model,
        env_name="GRAMGUARD_NLI_MODEL",
        repo_cache_name="models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli",
        display_name="NLI model",
    )
    cola_path = resolve_validation_model_path(
        args.cola_model,
        env_name="GRAMGUARD_COLA_MODEL",
        repo_cache_name="models--textattack--roberta-base-CoLA",
        display_name="CoLA model",
    )
    semantic_gate = SemanticEquivalenceGate(
        nli_path=nli_path,
        cola_path=cola_path,
        device=validation_device,
        min_entailment=args.min_bidirectional_entailment,
        max_contradiction=args.max_bidirectional_contradiction,
        min_cola_acceptability=args.min_cola_acceptability,
        max_cola_drop=args.max_cola_drop,
    )
    pos_tagger = OptionalPosTagger()
    if not pos_tagger.available:
        raise RuntimeError(f"Mandatory NLTK POS tagger unavailable: {pos_tagger.reason}")
    return {
        "model2": model2,
        "model4": model4,
        "rewrite_tokenizer": rewrite_tokenizer,
        "rewrite_model": rewrite_model,
        "rewrite_device": rewrite_device,
        "rewrite_model_spec": rewrite_spec,
        "semantic_model": semantic_model,
        "semantic_gate": semantic_gate,
        "pos_tagger": pos_tagger,
    }


def _write_llm_budget_snapshot(
    *,
    root: Path,
    budget: float,
    stem: str,
    text: str,
    original_text: str,
    initial_doc: Dict[str, float],
    final_doc: Dict[str, float],
    initial_null: Dict[str, float],
    final_null: Dict[str, float],
    selected_rows: List[Dict[str, Any]],
    stage_stop_reason: str,
    raw_path: Path,
    source: str,
    sample_index: int,
    rewrite_model_spec: str,
) -> Dict[str, Any]:
    label = _budget_label(budget)
    out_dir = root / f"budget_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"lexical_final_{stem}.txt"
    summary_path = out_dir / f"lexical_summary_{stem}.csv"
    selected_path = out_dir / f"lexical_selected_{stem}.csv"
    final_path.write_text(text, encoding="utf-8")
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    changed_count = word_diff_count(original_text, text)
    summary = {
        "mode": "llm_budget_curve",
        "rewrite_model": rewrite_model_spec,
        "budget": float(budget),
        "budget_percent": int(round(float(budget) * 100)),
        "input_file": str(raw_path),
        "source": source,
        "sample_index": int(sample_index),
        "word_count": int(word_count(original_text)),
        "selected_edit_count": int(len(selected_rows)),
        "applied_edit_count": int(len(selected_rows)),
        "changed_token_count": int(changed_count),
        "actual_edit_ratio": float(changed_count / max(1, word_count(original_text))),
        "initial_H4": float(initial_doc["H4"]),
        "initial_A24": float(initial_doc["A24"]),
        "final_H4": float(final_doc["H4"]),
        "final_A24": float(final_doc["A24"]),
        "initial_human_null_distance": float(initial_null["human_null_distance"]),
        "final_human_null_distance": float(final_null["human_null_distance"]),
        "human_null_distance_improvement": float(initial_null["human_null_distance"] - final_null["human_null_distance"]),
        "stop_reason": stage_stop_reason,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary


def run_llm_budget_curve_one(
    args: argparse.Namespace,
    *,
    raw_path: Path,
    texts: Sequence[str],
    sample_index: int,
    runtime: Dict[str, Any],
    human_calibration: pd.DataFrame,
    budgets: Sequence[float],
) -> List[Dict[str, Any]]:
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    condition = safe_name(raw_path, args.source)
    stem = f"{condition}_sample{sample_index}"
    selected_master = root / f"llm_budget_selected_{stem}.csv"
    candidates_master = root / f"llm_budget_candidates_{stem}.csv"
    summary_master = root / f"llm_budget_trajectory_summary_{stem}.csv"

    # Resume only when every requested snapshot is complete.
    resumed = []
    complete = True
    for budget in budgets:
        bdir = root / f"budget_{_budget_label(budget)}"
        sp = bdir / f"lexical_summary_{stem}.csv"
        fp = bdir / f"lexical_final_{stem}.txt"
        if not (sp.exists() and fp.exists() and fp.stat().st_size > 0):
            complete = False
            break
        resumed.append(pd.read_csv(sp).iloc[0].to_dict())
    if complete and len(resumed) == len(budgets):
        print(f"[RESUME] sample={sample_index} all LLM budget snapshots already complete")
        return resumed

    original_text = clean_text(texts[sample_index])
    working_text = original_text
    model2, model4 = runtime["model2"], runtime["model4"]
    initial_doc = score_document(model2, model4, original_text)
    initial_null = human_null_stats(
        length=word_count(original_text),
        a24=float(initial_doc["A24"]),
        calibration=human_calibration,
        rel_window=args.human_null_rel_window,
        min_neighbors=args.human_null_min_neighbors,
    )
    all_candidates: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    snapshot_summaries: List[Dict[str, Any]] = []

    print("\n" + "=" * 112)
    print("GramGuard-V3 FINAL | HRCG-Guided Instruct-LLM Human-Null Budget Curve")
    print("=" * 112)
    print(f"Sample                : {sample_index}")
    print(f"Words                 : {word_count(original_text)}")
    print(f"Rewrite model         : {runtime['rewrite_model_spec']}")
    print(f"Initial A24           : {initial_doc['A24']:.10f}")
    print(f"Initial |z| distance  : {initial_null['human_null_distance']:.6f}")
    print(f"Budgets               : {[int(b*100) for b in budgets]}%")

    global_round = 0
    for budget in budgets:
        stage_round = 0
        stage_stop = "budget ceiling reached"
        while stage_round < int(args.llm_max_rounds_per_budget):
            current_changed = word_diff_count(original_text, working_text)
            current_ratio = current_changed / max(1, word_count(original_text))
            if current_ratio >= float(budget) - float(args.budget_tolerance):
                stage_stop = "budget target reached"
                break
            remaining_words = max(1, int(math.floor(float(budget) * word_count(original_text))) - current_changed)
            desired_change = min(
                remaining_words,
                max(2, int(round(word_count(original_text) * float(args.llm_desired_change_fraction))))
            )

            current_doc = score_document(model2, model4, working_text)
            attribution = full_token_attribution(model2, model4, working_text)
            verify_attribution(current_doc, attribution, working_text)
            rng = random.Random(int(args.seed) + sample_index * 1009 + (global_round + 1) * 9173)
            targets, _, _ = select_span_targets(
                attribution,
                working_text,
                runtime["pos_tagger"],
                "hrcg",
                args.llm_target_count,
                args.min_word_length,
                set(),
                rng,
            )
            if targets.empty:
                stage_stop = "no eligible HRCG targets"
                break

            # Deduplicate HRCG targets that fall in the same sentence.
            current_words = clean_text(working_text).split()
            sentence_targets = []
            seen_bounds = set()
            for target in targets.itertuples(index=False):
                ss, se = sentence_bounds(current_words, int(target.position))
                bounds = (ss, se)
                if bounds in seen_bounds:
                    continue
                seen_bounds.add(bounds)
                sentence_targets.append((target, ss, se))
                if len(sentence_targets) >= int(args.llm_target_sentences):
                    break

            global_round += 1
            stage_round += 1
            round_rows: List[Dict[str, Any]] = []
            for target, ss, se in sentence_targets:
                raw_candidates, sent_start, sent_end, original_sentence = generate_llm_sentence_candidates(
                    text=working_text,
                    center_position=int(target.position),
                    tokenizer=runtime["rewrite_tokenizer"],
                    model=runtime["rewrite_model"],
                    device=runtime["rewrite_device"],
                    candidate_count=args.llm_candidates,
                    temperature=args.llm_temperature,
                    top_p=args.llm_top_p,
                    max_new_tokens=args.llm_max_new_tokens,
                    max_changed_words=max(2, remaining_words),
                    desired_changed_words=max(2, desired_change),
                )
                if not raw_candidates:
                    continue
                evaluated = evaluate_llm_budget_candidates(
                    original_document_text=original_text,
                    current_text=working_text,
                    target_row=pd.Series(target._asdict()),
                    raw_candidates=raw_candidates,
                    sentence_start=sent_start,
                    sentence_end=sent_end,
                    original_sentence=original_sentence,
                    model2=model2,
                    model4=model4,
                    semantic_model=runtime["semantic_model"],
                    semantic_gate=runtime["semantic_gate"],
                    current_doc=current_doc,
                    human_calibration=human_calibration,
                    human_rel_window=args.human_null_rel_window,
                    human_min_neighbors=args.human_null_min_neighbors,
                    min_sentence_similarity=args.llm_sentence_similarity,
                    min_document_similarity=args.min_document_similarity,
                    max_edit_ratio=float(budget),
                    min_human_gain=args.min_human_null_gain,
                )
                for row in evaluated:
                    row["global_round"] = int(global_round)
                    row["budget_stage"] = float(budget)
                    round_rows.append(row)
                    all_candidates.append(row)

            if not round_rows:
                stage_stop = "LLM produced no usable local rewrites"
                break
            round_df = pd.DataFrame(round_rows)
            best = _select_best_llm_candidate(round_df)
            valid_count = int(round_df["valid_candidate"].sum())
            print(
                f"  budget={int(budget*100):02d}% round={stage_round:02d} "
                f"ratio={current_ratio:.4f} remaining={remaining_words} "
                f"evaluated={len(round_df)} valid={valid_count}"
            )
            if best is None:
                stage_stop = "no human-null-improving LLM rewrite passed quality gates within budget"
                break

            record = best.to_dict()
            record["original_core"] = str(record.get("original_span", ""))
            record["candidate_core"] = str(record.get("candidate_span", ""))
            record["text_before"] = working_text
            record["text_after"] = str(best["candidate_text"])
            record["selected_for_round"] = True
            selected_rows.append(record)
            working_text = str(best["candidate_text"])
            new_changed = word_diff_count(original_text, working_text)
            new_ratio = new_changed / max(1, word_count(original_text))
            print(
                f"    SELECT {best['original_sentence']!r}\n"
                f"        -> {best['candidate_sentence']!r}\n"
                f"       sentence_changed={int(best['sentence_changed_word_count'])} "
                f"cumulative_changed={new_changed} ratio={new_ratio:.4f} "
                f"|z| {best['human_null_distance_before']:.4f}->{best['human_null_distance_after']:.4f} "
                f"gain={best['human_null_distance_improvement']:.4f}"
            )

        final_doc = score_document(model2, model4, working_text)
        final_null = human_null_stats(
            length=word_count(working_text),
            a24=float(final_doc["A24"]),
            calibration=human_calibration,
            rel_window=args.human_null_rel_window,
            min_neighbors=args.human_null_min_neighbors,
        )
        snap = _write_llm_budget_snapshot(
            root=root,
            budget=float(budget),
            stem=stem,
            text=working_text,
            original_text=original_text,
            initial_doc=initial_doc,
            final_doc=final_doc,
            initial_null=initial_null,
            final_null=final_null,
            selected_rows=selected_rows,
            stage_stop_reason=stage_stop,
            raw_path=raw_path,
            source=args.source,
            sample_index=sample_index,
            rewrite_model_spec=runtime["rewrite_model_spec"],
        )
        snapshot_summaries.append(snap)
        print(
            f"[LLM SNAPSHOT {int(budget*100):02d}%] edits={len(selected_rows)} "
            f"changed={snap['changed_token_count']} ratio={snap['actual_edit_ratio']:.4f} "
            f"human |z|={snap['final_human_null_distance']:.4f} stop={stage_stop}"
        )

    pd.DataFrame(all_candidates).to_csv(candidates_master, index=False)
    pd.DataFrame(selected_rows).to_csv(selected_master, index=False)
    pd.DataFrame(snapshot_summaries).to_csv(summary_master, index=False)
    return snapshot_summaries


def run_llm_budget_curve_batch(args: argparse.Namespace) -> None:
    if not args.allow_rewrite:
        raise RuntimeError("--mode llm_budget_curve requires --allow_rewrite")
    raw_path = Path(args.raw_file).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    texts = extract_texts(raw_path, args.source)
    indices = parse_index_spec(args.sample_indices) if args.sample_indices else [int(args.sample_index)]
    invalid = [i for i in indices if i < 0 or i >= len(texts)]
    if invalid:
        raise IndexError(f"Invalid sample indices: {invalid}; available={len(texts)}")
    budgets = sorted({float(x.strip()) for x in str(args.edit_budgets).split(",") if x.strip()})
    if not budgets or any(b <= 0 or b > 0.60 for b in budgets):
        raise ValueError("--edit_budgets must contain fractions in (0, 0.60]")

    runtime = _load_llm_budget_runtime(args)
    human_calibration = load_human_null_calibration(Path(args.human_null_csv))
    all_by_budget: Dict[float, List[Dict[str, Any]]] = {b: [] for b in budgets}
    for n, sample_index in enumerate(indices, start=1):
        print(f"\n{'#' * 112}\nFINAL LLM BUDGET CURVE {n}/{len(indices)} | sample={sample_index}\n{'#' * 112}")
        snapshots = run_llm_budget_curve_one(
            args,
            raw_path=raw_path,
            texts=texts,
            sample_index=sample_index,
            runtime=runtime,
            human_calibration=human_calibration,
            budgets=budgets,
        )
        for rec in snapshots:
            all_by_budget[float(rec["budget"])].append(rec)

    rows = []
    root = Path(args.output_dir).expanduser().resolve()
    for budget in budgets:
        frame = pd.DataFrame(all_by_budget[budget])
        if not frame.empty:
            frame = frame.sort_values("sample_index")
        bdir = root / f"budget_{_budget_label(budget)}"
        bdir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(bdir / "batch_generation_summary.csv", index=False)
        edited = int((frame["changed_token_count"] > 0).sum()) if not frame.empty else 0
        reached = int((frame["actual_edit_ratio"] >= max(0.0, budget - args.budget_tolerance)).sum()) if not frame.empty else 0
        rows.append({
            "budget": float(budget),
            "budget_percent": int(round(budget * 100)),
            "documents": int(len(frame)),
            "edited_documents": edited,
            "coverage": float(edited / len(frame)) if len(frame) else float("nan"),
            "documents_reaching_budget": reached,
            "budget_reach_rate": float(reached / len(frame)) if len(frame) else float("nan"),
            "mean_actual_edit_ratio": float(frame["actual_edit_ratio"].mean()) if len(frame) else float("nan"),
            "median_actual_edit_ratio": float(frame["actual_edit_ratio"].median()) if len(frame) else float("nan"),
            "mean_human_null_distance_before": float(frame["initial_human_null_distance"].mean()) if len(frame) else float("nan"),
            "mean_human_null_distance_after": float(frame["final_human_null_distance"].mean()) if len(frame) else float("nan"),
            "mean_human_null_distance_improvement": float(frame["human_null_distance_improvement"].mean()) if len(frame) else float("nan"),
        })
    master = pd.DataFrame(rows)
    out_path = Path(args.output_dir).expanduser().resolve() / "llm_budget_curve_generation_master.csv"
    master.to_csv(out_path, index=False)
    print("\n" + "=" * 112)
    print("FINAL LLM BUDGET-CURVE GENERATION SUMMARY")
    print("=" * 112)
    print(master.to_string(index=False))
    print(f"\nSaved: {out_path}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GramGuard-V3 lexical audit plus detector-independent masked-span reconstruction."
    )
    parser.add_argument(
        "--mode",
        choices=["discover", "audit", "rewrite", "span_rewrite", "budget_curve", "llm_budget_curve", "evaluate"],
        default="audit",
    )
    parser.add_argument("--allow_rewrite", action="store_true")
    parser.add_argument("--raw_file", default="")
    parser.add_argument("--source", choices=["ai", "human"], default="ai")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sample_indices", default="", help="Batch indices for span_rewrite, e.g. 0-19")
    parser.add_argument("--mlm_model", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nli_model", default="auto")
    parser.add_argument("--cola_model", default="auto")
    parser.add_argument("--validation_device", default="cpu")
    parser.add_argument("--min_bidirectional_entailment", type=float, default=0.98)
    parser.add_argument("--max_bidirectional_contradiction", type=float, default=0.05)
    parser.add_argument("--min_cola_acceptability", type=float, default=0.80)
    parser.add_argument("--max_cola_drop", type=float, default=0.02)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR / "gramguard_v3"))
    parser.add_argument("--seed", type=int, default=42)

    # Existing lexical mode.
    parser.add_argument("--target_count", type=int, default=5)
    parser.add_argument("--mlm_candidates", type=int, default=30)
    parser.add_argument("--max_edits", type=int, default=1)
    parser.add_argument("--local_radius", type=int, default=4)
    parser.add_argument("--max_context_words", type=int, default=80)
    parser.add_argument("--min_word_length", type=int, default=3)
    parser.add_argument("--local_context_semantic_threshold", type=float, default=0.97)
    parser.add_argument("--sentence_semantic_threshold", type=float, default=0.98)
    parser.add_argument("--max_mlm_rank", type=int, default=10)
    parser.add_argument("--max_mlm_global_rank", type=int, default=500)
    parser.add_argument("--min_mlm_probability", type=float, default=1e-5)
    parser.add_argument("--min_local_h4_gain", type=float, default=0.0)
    parser.add_argument("--max_document_h4_increase", type=float, default=0.0)
    parser.add_argument("--display_candidates", type=int, default=40)

    # Span reconstruction mode. Selector is the only controlled experimental variable.
    parser.add_argument("--span_selector", choices=["hrcg", "h4", "random"], default="hrcg")
    parser.add_argument("--span_radius", type=int, default=1)
    parser.add_argument("--span_target_count", type=int, default=6)
    parser.add_argument("--span_candidates", type=int, default=8)
    parser.add_argument("--span_branch_k", type=int, default=6)
    parser.add_argument("--span_beam_width", type=int, default=16)
    parser.add_argument("--max_span_edits", type=int, default=4)
    parser.add_argument("--max_edit_ratio", type=float, default=0.12)
    parser.add_argument("--span_local_similarity", type=float, default=0.90)
    parser.add_argument("--span_sentence_similarity", type=float, default=0.96)
    parser.add_argument("--span_min_pos_agreement", type=float, default=0.67)
    parser.add_argument("--span_min_mlm_probability", type=float, default=1e-7)

    # Final Idea-3 budget-curve mode.
    parser.add_argument(
        "--human_null_csv",
        default=str(ROOT / "assets" / "umbc_human_null_antique_features_10000.csv"),
    )
    parser.add_argument("--human_null_rel_window", type=float, default=0.25)
    parser.add_argument("--human_null_min_neighbors", type=int, default=100)
    parser.add_argument("--edit_budgets", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--require_final_budgets", action="store_true")
    parser.add_argument("--max_rounds_per_budget", type=int, default=10)
    parser.add_argument("--min_document_similarity", type=float, default=0.90)

    # Final local instruct-LLM rewrite engine.
    parser.add_argument("--rewrite_model", default="auto")
    parser.add_argument("--allow_model_download", action="store_true")
    parser.add_argument("--rewrite_device", default="auto")
    parser.add_argument("--llm_candidates", type=int, default=10)
    parser.add_argument("--llm_target_count", type=int, default=12)
    parser.add_argument("--llm_target_sentences", type=int, default=6)
    parser.add_argument("--llm_temperature", type=float, default=0.90)
    parser.add_argument("--llm_top_p", type=float, default=0.95)
    parser.add_argument("--llm_max_new_tokens", type=int, default=128)
    parser.add_argument("--llm_max_rounds_per_budget", type=int, default=8)
    parser.add_argument("--llm_desired_change_fraction", type=float, default=0.035)
    parser.add_argument("--llm_sentence_similarity", type=float, default=0.86)
    parser.add_argument("--min_human_null_gain", type=float, default=0.0)
    parser.add_argument("--budget_tolerance", type=float, default=0.0125)

    # Detector evaluation.
    parser.add_argument("--attack_output_dir", default="outputs_gramguard_v3")
    parser.add_argument("--eval_output_dir", default="outputs_gramguard_v3_detector")
    parser.add_argument("--eval_indices", default="0-4")
    parser.add_argument("--causal_model", default="auto")
    parser.add_argument("--causal_device", default="cpu")
    parser.add_argument("--eval_max_length", type=int, default=768)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(int(args.seed))

    if args.mode == "discover":
        discovered = discover_local_masked_lms()
        print(discovered.to_string(index=False) if not discovered.empty else "No local masked LM found")
        return
    if args.mode == "evaluate":
        if not args.raw_file:
            raise RuntimeError("--raw_file is required for --mode evaluate")
        run_paired_evaluation(args)
        return
    if args.mode == "llm_budget_curve":
        if args.source != "ai":
            raise RuntimeError("llm_budget_curve is intended for AI text")
        run_llm_budget_curve_batch(args)
        return
    if args.mode == "budget_curve":
        if args.source != "ai":
            raise RuntimeError("budget_curve is intended for AI text")
        run_budget_curve_batch(args)
        return
    if args.mode == "span_rewrite":
        if args.source != "ai":
            raise RuntimeError("span_rewrite is intended for AI text")
        run_span_rewrite_batch(args)
        return
    if args.mode == "rewrite" and not args.allow_rewrite:
        raise RuntimeError("--mode rewrite requires --allow_rewrite")
    if args.source != "ai" and args.mode == "rewrite":
        raise RuntimeError("Rewrite mode is intended for AI text; use audit for human controls")
    run_audit_or_rewrite(args)


if __name__ == "__main__":
    main()
