#!/usr/bin/env bash
#SBATCH --job-name=verify_loop
#SBATCH --partition=shire-general
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --output=/home/zhenwu/isomorphism_pred_KBQA/agentic_clean/logs/%x_%j.out
#SBATCH --mem=200GB
#SBATCH --time=2-00:00:00

set -euo pipefail
cd /home/zhenwu/isomorphism_pred_KBQA

# ── What to run ───────────────────────────────────────────────────────────────
DATASET="webqsp"        # webqsp | grailqa | grailqapp
VERIFIER="double"           # none | exe | double
FEEDBACK="iso"         # plain | iso
SHOTMODE="fs"            # fs | zs

# ── Predicted ISO (optional) ──────────────────────────────────────────────────
# Set to "true" to use predicted ISO (restricts to correct==false examples).
# Leave empty/false to use gold ISO (normal double mode).
USE_PRED_ISO="true"
# USE_PRED_ISO="false"

# ── Model / provider ─────────────────────────────────────────────────────────
# PROVIDER="openai"      ; MODEL="gpt-5-mini"
# PROVIDER="claude"      ; MODEL="claude-opus-4-6"
# PROVIDER="claude"      ; MODEL="claude-sonnet-4-6"
# PROVIDER="gemini"      ; MODEL="gemini-2.0-flash-exp"
# PROVIDER="vllm_server" ; MODEL="glm-4.7-fp8"
# PROVIDER="vllm_server" ; MODEL="google/gemma-3-27b-it"
# PROVIDER="vllm_server" ; MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# PROVIDER="vllm_server" ; MODEL="Qwen/Qwen3-14B"
# PROVIDER="vllm_server", MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
# PROVIDER="huggingface" ; MODEL="google/gemma-3-27b-it"
# PROVIDER="openrouter"  ; MODEL="minimax/minimax-m2.5"
PROVIDER="openrouter"  ; MODEL="moonshotai/kimi-k2.5"
TEMPERATURE="0.6"          # leave empty to use each model's default
MAX_ATTEMPTS=10
MAX_NEW_TOKENS=128
TENSOR_PARALLEL_SIZE=2
MAX_LENGTH=131072        # vllm only; auto-detected from tokenizer if unset
# MAX_LENGTH=65536   
# MAX_LENGTH=32768          # set lower for faster iteration during testing; increase for final runs    
# MAX_LENGTH=128000        # set lower for faster iteration during testing; increase for final runs 
API_BASE="http://localhost:8000/v1"
API_KEY="EMPTY"

# ── API keys ──────────────────────────────────────────────────────────────────
export ANTHROPIC_API_KEY=""   # needed for claude provider
# export OPENAI_API_KEY=""    # needed for openai provider
export OPENROUTER_API_KEY=""

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR="agentic_clean/results-max_attempts10"
# OUT_DIR="agentic_clean/results_rerun"

# ── Dataset-specific paths (auto-selected) ────────────────────────────────────
if   [[ "$DATASET" == "webqsp" ]]; then
    DATA_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/webqsp/KBQA_walklet_edge_label_768_dim_data.dill"
    TEMPLATE="agentic_clean/prompts/user_prompt.txt"
    PRED_ISO_FILE="agentic_clean/pred_iso/webqsp_unified_vote_1343.json"
    if [[ "$FEEDBACK" == "iso" ]]; then
        SYS_PROMPT="agentic_clean/prompts/fs_sys_prompt_webqsp_iso_verifier.txt"
    else
        SYS_PROMPT="agentic_clean/prompts/fs_sys_prompt_webqsp.txt"
    fi
elif [[ "$DATASET" == "grailqa" ]]; then
    DATA_PATH="/home/zhenwu/isomorphism_pred_KBQA/agentic/grailqa_dev_sampled_1000.dill"
    TEMPLATE="agentic_clean/prompts/user_prompt_grailqa.txt"
    PRED_ISO_FILE="agentic_clean/pred_iso/grailqa_dev1000_Meta-Llama-3-8B-Instruct_ctx1024_majority_vote.json"
    if [[ "$FEEDBACK" == "iso" ]]; then
        SYS_PROMPT="agentic_clean/prompts/fs_sys_prompt_grailqa_iso_verifier.txt"
    else
        SYS_PROMPT="agentic_clean/prompts/fs_sys_prompt_grailqa.txt"
    fi
elif [[ "$DATASET" == "grailqapp" ]]; then
    DATA_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill"
    TEMPLATE="agentic_clean/prompts/user_prompt_grailqa.txt"
    PRED_ISO_FILE="agentic_clean/pred_iso/grailqa_test_grailqapp_llm_fs_iso_results_gpt-5-mini_5shot.json"
    if [[ "$FEEDBACK" == "iso" ]]; then
        SYS_PROMPT="agentic_clean/prompts/fs_sys_prompt_grailqa_iso_verifier.txt"
    else
        SYS_PROMPT="agentic_clean/prompts/fs_sys_prompt_grailqa.txt"
    fi
fi

PRED_ISO_PATH=""
[[ "$USE_PRED_ISO" == "true" ]] && PRED_ISO_PATH="$PRED_ISO_FILE"

# ── Run ───────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1 python -m agentic_clean.verify_loop \
    --dataset              "$DATASET"              \
    --data_path            "$DATA_PATH"            \
    --sys_prompt           "$SYS_PROMPT"           \
    --template             "$TEMPLATE"             \
    --provider             "$PROVIDER"             \
    --model                "$MODEL"                \
    --shotmode             "$SHOTMODE"             \
    --verifier             "$VERIFIER"             \
    --feedback             "$FEEDBACK"             \
    ${TEMPERATURE:+--temperature "$TEMPERATURE"} \
    --max_attempts         "$MAX_ATTEMPTS"         \
    --max_new_tokens       "$MAX_NEW_TOKENS"       \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_length           "$MAX_LENGTH"           \
    --api_base             "$API_BASE"             \
    --api_key              "$API_KEY"              \
    --anthropic_api_key    "$ANTHROPIC_API_KEY"    \
    ${PRED_ISO_PATH:+--pred_iso_path "$PRED_ISO_PATH"} \
    --out_dir              "$OUT_DIR"