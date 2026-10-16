# AnchorWrite

**Detector-Query-Free Rewriting of LLM-Generated Text via Human-Reference Alignment**

AnchorWrite is a detector-query-free framework for controlled rewriting of LLM-generated text using fixed human-reference linguistic evidence.

Attacked detectors are never queried during rewriting.

## Method

AnchorWrite combines:

- Human-reference KenLM 2-gram and 4-gram language models.
- Token-level cross-order evidence for revision localization.
- Document-level alignment with a fixed human-only null distribution.
- Semantic, grammatical, and structural preservation constraints.
- Four frozen rewriting regimes, L1 through L4.

Candidate selection minimizes the absolute distance between document-level A24 and the length-conditioned human-only reference distribution.

## Repository Structure

- anchorwrite/: Main rewriting implementation.
- src/: Human-reference scoring utilities.
- configs/: Portable configuration.
- scripts/: Frozen Qwen3 generation runners.
- evaluation/: Quality and detector evaluation scripts.
- assets/: Fixed human-reference calibration.
- data/: Processed evaluation data and detector scores.
- results/: Frozen paper results and analysis.

Historical GramGuard identifiers are retained internally where necessary for reproducibility.

## Experimental Setup

The primary backbone is Qwen3-8B.

The main benchmark contains 900 documents across six domains, with 150 documents per domain.

The frozen generation procedure uses the llm_budget_curve mode.

All attacked-detector evaluations are performed after rewriting outputs are frozen.

## Installation

The recorded primary environment uses:

- Python 3.10
- PyTorch 2.5.1+cu121
- Transformers 4.51.3

Install the Python dependencies using:

    python -m pip install -r requirements.txt

The requirements file is a preliminary reproduction specification; external checkpoints and the recorded CUDA environment are required for faithful reproduction.

## Human-Reference KenLM Models

Two fixed language models trained on cleaned UMBC human text are required:

- umbc_2gram.binary (approximately 3.8 GB)
- umbc_4gram.trie.binary (approximately 24 GB)

Frozen SHA-256 hashes:

2-gram:
b6dcd26d3dcf6bbc12b5e3b06f1dd8a4f103a940a831a33c929da9632f94a876

4-gram:
7e504019608e58d4f44dbc07a53acf489d4555aaf092205c833ace21732cd245

The Hugging Face model download link will be added once the frozen model binaries are uploaded and redistribution permissions are verified.

The fixed human-only calibration table is included in assets/.

KenLM paths can be provided through ANCHORWRITE_KENLM_2 and ANCHORWRITE_KENLM_4.

## Generation

Frozen generation runners:

- scripts/run_qwen3_L1_L2_L3_150.sh
- scripts/run_qwen3_L4_domain_neogpu.sh

External model checkpoints must be configured before generation.

## Evaluation

The repository includes processed evaluation inputs, frozen per-sample detector scores, and paper-facing results.

Quality evaluations include perplexity, BERTScore, bidirectional NLI diagnostics, semantic similarity, and lexical analysis.

Detector evaluation includes AUROC and AUPRC analyses.

The strict bootstrap configuration uses:

- 2,000 bootstrap replicates
- Seed 20260917
- Canonical AUROC tolerance 0.0005

Frozen paper results are stored in results/paper/.

New reproduction outputs are written to results/reproduced/.

Historical detector evaluation wrappers may use score-orientation procedures different from the strict canonical bootstrap protocol.

## Reproducibility and Data

The repository preserves the frozen experimental configurations and processed evaluation records.

Exact generation and detector inference additionally require external checkpoints and the corresponding official detector implementations.

Dataset and model redistribution remain subject to the relevant source licenses and permissions.

## Citation

Citation details will be added following the anonymous review period.
