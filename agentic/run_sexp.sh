#!/usr/bin/env bash
#SBATCH --job-name=sexp_prompting
#SBATCH --partition=shire-general
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --output=/home/zhenwu/isomorphism_pred_KBQA/agentic_clean/logs/%x_%j.out
#SBATCH --mem=200GB
#SBATCH --time=1-00:00:00

set -euo pipefail
cd /home/zhenwu/isomorphism_pred_KBQA

# ── What to run ────────────────────────────────────────────────────────────────
# DATASET="grailqapp"     # webqsp | grailqa | grailqapp
# DATASET="grailqa"     # webqsp | grailqa | grailqapp
DATASET="webqsp"        # webqsp | grailqa | grailqapp

# ISO hint mode:
#   none      – no hint (baseline)
#   gold      – hint computed from gold s-expression (oracle upper bound)
#   predicted – hint loaded from --iso_pred_path (realistic setting)
# ISO_MODE="none"
# ISO_MODE="gold"
ISO_MODE="predicted"

# ── Model / provider ──────────────────────────────────────────────────────────
# PROVIDER="openai"      ; MODEL="gpt-4o"
# PROVIDER="openai"      ; MODEL="gpt-5-mini"
# PROVIDER="claude"      ; MODEL="claude-sonnet-4-6"
# PROVIDER="openrouter"  ; MODEL="moonshotai/kimi-k2.5"
# PROVIDER="openrouter"  ; MODEL="minimax/minimax-m2.5"
# PROVIDER="gemini"      ; MODEL="gemini-2.0-flash-exp"
# PROVIDER="vllm_server" ; MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
PROVIDER="vllm_server" ; MODEL="Qwen/Qwen3-14B"
# PROVIDER="vllm_server" ; MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
# PROVIDER="huggingface" ; MODEL="google/gemma-3-27b-it"
# PROVIDER="openrouter"  ; MODEL="moonshotai/kimi-k2.5"

# replace / in MODEL with _ for filename
MODEL_FILENAME=$(echo "$MODEL" | tr '/' '_')

RUN_NAME="${DATASET}_${MODEL_FILENAME}_fs_sexp_${ISO_MODE}"
# RUN_NAME="${DATASET}_${MODEL_FILENAME}_fs_sexp_${ISO_MODE}-1"
# RUN_NAME="${DATASET}_${MODEL_FILENAME}_fs_sexp_${ISO_MODE}-2"
# RUN_NAME="${DATASET}_${MODEL_FILENAME}_fs_sexp_${ISO_MODE}-3"
# RUN_NAME="${DATASET}_${MODEL_FILENAME}_fs_sexp_${ISO_MODE}-4"
# RUN_NAME="${DATASET}_${MODEL_FILENAME}_fs_sexp_${ISO_MODE}-5"


TEMPERATURE="0.7"
MAX_NEW_TOKENS=128
TENSOR_PARALLEL_SIZE=2
# MAX_LENGTH=131072        # vllm only
# MAX_LENGTH=65536          # set lower for faster iteration during testing; increase for final runs
MAX_LENGTH=32768          # set lower for faster iteration during testing; increase for final runs
API_BASE="http://localhost:8000/v1"
API_KEY="EMPTY"

BATCH_SIZE=500

# ── API keys ──────────────────────────────────────────────────────────────────
export OPENROUTER_API_KEY=""
# export OPENAI_API_KEY=""
# export ANTHROPIC_API_KEY=""

# ── Dataset-specific paths ────────────────────────────────────────────────────
if   [[ "$DATASET" == "webqsp" ]]; then
    DATA_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/webqsp/KBQA_walklet_edge_label_768_dim_data.dill"
    ISO_PRED_FILE="agentic_clean/pred_iso/webqsp_unified_vote_1343.json"

elif [[ "$DATASET" == "grailqa" ]]; then
    DATA_PATH="/home/zhenwu/isomorphism_pred_KBQA/agentic/grailqa_dev_sampled_1000.dill"
    ISO_PRED_FILE="agentic_clean/pred_iso/grailqa_dev1000_Meta-Llama-3-8B-Instruct_ctx1024_majority_vote.json"

elif [[ "$DATASET" == "grailqapp" ]]; then
    DATA_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill"
    ISO_PRED_FILE="agentic_clean/pred_iso/grailqa_test_grailqapp_llm_fs_iso_results_gpt-5-mini_5shot.json"
fi

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR="agentic_clean/results_sexp"

# ── Run ───────────────────────────────────────────────────────────────────────
ISO_PRED_ARG=""
[[ "$ISO_MODE" == "predicted" ]] && ISO_PRED_ARG="--iso_pred_path $ISO_PRED_FILE"

CUDA_VISIBLE_DEVICES=0,1 python -m agentic_clean.sexp_prompting \
    --dataset              "$DATASET"              \
    --data_path            "$DATA_PATH"            \
    --provider             "$PROVIDER"             \
    --model                "$MODEL"                \
    --iso_mode             "$ISO_MODE"             \
    ${ISO_PRED_ARG}                                \
    ${TEMPERATURE:+--temperature "$TEMPERATURE"}   \
    --max_new_tokens       "$MAX_NEW_TOKENS"       \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_length           "$MAX_LENGTH"           \
    --api_base             "$API_BASE"             \
    --api_key              "$API_KEY"              \
    --openrouter_api_key   "$OPENROUTER_API_KEY"   \
    --anthropic_api_key    "$ANTHROPIC_API_KEY"    \
    --out_dir              "$OUT_DIR"       \
    --run_name             "$RUN_NAME"      \
    ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"}
