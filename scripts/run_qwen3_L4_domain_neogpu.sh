#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 GPU_ID DOMAIN"
    exit 1
fi

GPU_ID="$1"
NAME="$2"

PROJECT="${ANCHORWRITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${ANCHORWRITE_OUTPUT_DIR:-$PROJECT/outputs/qwen3_150}"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRAMGUARD_SAMPLE_COOLDOWN=0

# Four concurrent validation pipelines share 28 vCPUs.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

: "${GRAMGUARD_REWRITE_MODEL:?Set GRAMGUARD_REWRITE_MODEL to the local Qwen3-8B directory}"
: "${GRAMGUARD_NLI_MODEL:?Set GRAMGUARD_NLI_MODEL to the local DeBERTa NLI directory}"
: "${GRAMGUARD_COLA_MODEL:?Set GRAMGUARD_COLA_MODEL to the local CoLA model directory}"

REWRITE_MODEL="$GRAMGUARD_REWRITE_MODEL"
NLI_MODEL="$GRAMGUARD_NLI_MODEL"
COLA_MODEL="$GRAMGUARD_COLA_MODEL"
HUMAN_NULL="${ANCHORWRITE_HUMAN_NULL:-$PROJECT/assets/umbc_human_null_antique_features_10000.csv}"
DATA_DIR="${ANCHORWRITE_DATA_DIR:-$PROJECT/data}"

declare -A FILES=(
  [social_sciences]="$DATA_DIR/gramguard_table6_150/social_sciences_moonshot_150.raw_data.json"
  [stem]="$DATA_DIR/gramguard_table6_150/stem_moonshot_150.raw_data.json"
  [humanities]="$DATA_DIR/gramguard_table6_150/humanities_moonshot_150.raw_data.json"
  [essay]="$DATA_DIR/gramguard_table6_150/essay_chatgpt_150.raw_data.json"
  [reuters]="$DATA_DIR/gramguard_table6_150/reuters_chatgpt_150.raw_data.json"
  [wp]="$DATA_DIR/gramguard_table6_150/wp_chatgpt_150.raw_data.json"
)

if [[ -z "${FILES[$NAME]+x}" ]]; then
    echo "Unknown domain: $NAME"
    exit 1
fi

OUT="$BASE_OUT/L4/$NAME"
LOG="$BASE_OUT/logs/L4_${NAME}.log"
DONE="$OUT/.DONE"

mkdir -p "$OUT"
mkdir -p "$BASE_OUT/logs"

echo
echo "===================================================================================================="
echo "L4 DOMAIN: $NAME | PHYSICAL GPU: $GPU_ID"
echo "===================================================================================================="

if [[ -f "$DONE" ]]; then
    echo "SKIP: $NAME already complete."
    exit 0
fi

START=0

if [[ -f "$LOG" ]]; then
    LAST_COMPLETED=$(
        awk '
          /^Sample[[:space:]]*:/ {sample=$3}
          /\[LLM SNAPSHOT 60%\]/ {last=sample}
          END {
              if (last != "") print last
          }
        ' "$LOG"
    )

    if [[ -n "${LAST_COMPLETED:-}" ]]; then
        START=$((LAST_COMPLETED + 1))
    fi
fi

if (( START >= 150 )); then
    echo "All 150 samples already appear complete."
    touch "$DONE"
    exit 0
fi

echo "Resume sample     : $START"
echo "Final sample      : 149"
echo "Budgets           : 30%, 40%, 50%, 60%"
echo "Candidates        : 40"
echo "Target count      : 40"
echo "Target sentences  : 12"
echo "Temperature       : 0.85"
echo "Top-p             : 0.95"
echo "Rounds/budget     : 20"
echo "Desired change    : 0.30"
echo "Sentence sim      : 0.68"
echo "Document sim      : 0.74"
echo "Entailment        : 0.70"
echo "Contradiction     : 0.25"
echo "CoLA              : 0.25"
echo "Max CoLA drop     : 0.45"
echo "Tolerance         : 0.04"
echo "Started           : $(date)"
echo "===================================================================================================="

python -u anchorwrite/anchorwrite.py \
  --mode llm_budget_curve \
  --allow_rewrite \
  --raw_file "${FILES[$NAME]}" \
  --source ai \
  --sample_indices "${START}-149" \
  --edit_budgets "0.30,0.40,0.50,0.60" \
  --rewrite_model "$REWRITE_MODEL" \
  --rewrite_device cuda \
  --nli_model "$NLI_MODEL" \
  --cola_model "$COLA_MODEL" \
  --human_null_csv "$HUMAN_NULL" \
  --validation_device cpu \
  --llm_candidates 40 \
  --llm_target_count 40 \
  --llm_target_sentences 12 \
  --llm_temperature 0.85 \
  --llm_top_p 0.95 \
  --llm_max_new_tokens 256 \
  --llm_max_rounds_per_budget 20 \
  --llm_desired_change_fraction 0.30 \
  --llm_sentence_similarity 0.68 \
  --min_document_similarity 0.74 \
  --min_bidirectional_entailment 0.70 \
  --max_bidirectional_contradiction 0.25 \
  --min_cola_acceptability 0.25 \
  --max_cola_drop 0.45 \
  --budget_tolerance 0.04 \
  --output_dir "$OUT" \
  2>&1 | tee -a "$LOG"

touch "$DONE"

echo
echo "===================================================================================================="
echo "FINISHED: L4 / $NAME / GPU $GPU_ID"
echo "Time: $(date)"
echo "===================================================================================================="
