#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ANCHORWRITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export GRAMGUARD_SAMPLE_COOLDOWN=5
export GRAMGUARD_DOMAIN_COOLDOWN=60

BASE_OUT="${ANCHORWRITE_OUTPUT_DIR:-$PROJECT/outputs/qwen3_150}"
mkdir -p "$BASE_OUT/logs"

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

DOMAINS=(
  social_sciences
  stem
  humanities
  essay
  reuters
  wp
)

run_job () {
    LEVEL="$1"
    NAME="$2"

    OUT="$BASE_OUT/$LEVEL/$NAME"
    LOG="$BASE_OUT/logs/${LEVEL}_${NAME}.log"
    DONE="$OUT/.DONE"

    if [[ -f "$DONE" ]]; then
        echo "===================================================================================================="
        echo "SKIP: $LEVEL / $NAME already completed"
        echo "===================================================================================================="
        return
    fi

    # Remove only an incomplete previous attempt.
    if [[ -d "$OUT" ]]; then
        echo "Removing incomplete output: $OUT"
        rm -rf "$OUT"
    fi

    mkdir -p "$OUT"

    case "$LEVEL" in

      L1)
        BUDGETS="0.05,0.10,0.15,0.20"
        CANDIDATES=10
        TARGET_COUNT=12
        TARGET_SENTENCES=6
        TEMP=0.7
        ROUNDS=8
        DESIRED=0.035
        SENT_SIM=0.86
        DOC_SIM=0.90
        ENTAIL=0.90
        CONTRA=0.10
        COLA=0.50
        COLA_DROP=0.25
        TOL=0.0125
        ;;

      L2)
        BUDGETS="0.05,0.10,0.15,0.20,0.25"
        CANDIDATES=20
        TARGET_COUNT=20
        TARGET_SENTENCES=8
        TEMP=0.7
        ROUNDS=12
        DESIRED=0.07
        SENT_SIM=0.80
        DOC_SIM=0.85
        ENTAIL=0.85
        CONTRA=0.15
        COLA=0.40
        COLA_DROP=0.35
        TOL=0.02
        ;;

      L3)
        BUDGETS="0.20,0.30,0.40,0.50"
        CANDIDATES=30
        TARGET_COUNT=30
        TARGET_SENTENCES=10
        TEMP=0.8
        ROUNDS=15
        DESIRED=0.20
        SENT_SIM=0.72
        DOC_SIM=0.78
        ENTAIL=0.75
        CONTRA=0.20
        COLA=0.30
        COLA_DROP=0.40
        TOL=0.03
        ;;

      *)
        echo "Unknown level: $LEVEL"
        exit 1
        ;;
    esac

    echo
    echo "####################################################################################################"
    echo "STARTING $LEVEL / $NAME"
    echo "Time: $(date)"
    echo "Input: ${FILES[$NAME]}"
    echo "Output: $OUT"
    echo "Budgets: $BUDGETS"
    echo "####################################################################################################"

    CUDA_VISIBLE_DEVICES=0 python -u anchorwrite/anchorwrite.py \
      --mode llm_budget_curve \
      --allow_rewrite \
      --raw_file "${FILES[$NAME]}" \
      --source ai \
      --sample_indices "0-149" \
      --edit_budgets "$BUDGETS" \
      --rewrite_model "$REWRITE_MODEL" \
      --rewrite_device cuda \
      --nli_model "$NLI_MODEL" \
      --cola_model "$COLA_MODEL" \
      --human_null_csv "$HUMAN_NULL" \
      --validation_device cpu \
      --llm_candidates "$CANDIDATES" \
      --llm_target_count "$TARGET_COUNT" \
      --llm_target_sentences "$TARGET_SENTENCES" \
      --llm_temperature "$TEMP" \
      --llm_top_p 0.95 \
      --llm_max_new_tokens 256 \
      --llm_max_rounds_per_budget "$ROUNDS" \
      --llm_desired_change_fraction "$DESIRED" \
      --llm_sentence_similarity "$SENT_SIM" \
      --min_document_similarity "$DOC_SIM" \
      --min_bidirectional_entailment "$ENTAIL" \
      --max_bidirectional_contradiction "$CONTRA" \
      --min_cola_acceptability "$COLA" \
      --max_cola_drop "$COLA_DROP" \
      --budget_tolerance "$TOL" \
      --output_dir "$OUT" \
      2>&1 | tee "$LOG"

    touch "$DONE"

    echo "[COOLDOWN] Domain complete; sleeping ${GRAMGUARD_DOMAIN_COOLDOWN}s"
    sleep "$GRAMGUARD_DOMAIN_COOLDOWN"

    echo
    echo "####################################################################################################"
    echo "FINISHED $LEVEL / $NAME"
    echo "Time: $(date)"
    echo "####################################################################################################"
}

echo "===================================================================================================="
echo "GramGuard-V3 Qwen3-8B MASTER RUN"
echo "Started: $(date)"
echo "Python : $(command -v python)"
echo "Output : $BASE_OUT"
echo "===================================================================================================="

for LEVEL in L1 L2 L3
do
    for NAME in "${DOMAINS[@]}"
    do
        run_job "$LEVEL" "$NAME"
    done
done

echo
echo "===================================================================================================="
echo "ALL L1 + L2 + L3 RUNS COMPLETED"
echo "Finished: $(date)"
echo "===================================================================================================="
