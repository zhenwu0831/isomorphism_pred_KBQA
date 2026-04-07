#!/usr/bin/env bash
# Run self-consistency selection across none-1 to none-5 runs.
#
# For each dataset, for the specified model:
#   1. Majority vote among none-1..5 answers
#   2. If no majority → LLM judge (same model) picks best candidate
#   3. Saves {dataset}_{model}_fs_sexp_SC.json to results_sexp
#
# Usage:
#   bash agentic_clean/run_self_consistency.sh

set -euo pipefail
cd /home/zhenwu/isomorphism_pred_KBQA

# ── Model (runs all three datasets for this model) ────────────────────────────
# MODEL="moonshotai_kimi-k2.5"
# MODEL="gpt-5-mini"
# MODEL="claude-sonnet-4-6"
MODEL="Qwen_Qwen3-14B"
# MODEL="Qwen_Qwen3-30B-A3B-Instruct-2507"

# ── Datasets to run ───────────────────────────────────────────────────────────
DATASETS=("webqsp" "grailqa" "grailqapp")

# ── Overwrite: re-run even if SC file already exists ──────────────────────────
# OVERWRITE="--overwrite"
OVERWRITE=""

SEED=42

# ── API keys ──────────────────────────────────────────────────────────────────
export OPENROUTER_API_KEY=""
export ANTHROPIC_API_KEY=""
# export OPENAI_API_KEY=""

# ── Run ───────────────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    echo "=== $DATASET / $MODEL ==="
    /home/zhenwu/anaconda3/envs/kbqa/bin/python -u -m agentic_clean.self_consistency \
        --dataset "$DATASET" \
        --model   "$MODEL"   \
        ${OVERWRITE}         \
        --seed    "$SEED"
done
