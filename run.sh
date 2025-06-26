#!/usr/bin/env bash
#SBATCH --job-name=ISO_KBQA
#SBATCH --partition=shire-general
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --output=/home/zhenwu/isomorphism_pred_KBQA/logs/%x_%j.out
#SBATCH --mem=80GB
#SBATCH --time=2-00:00:00

set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
MODEL_TYPE="cod"                   # text, graph, or cod
BACKBONE="t5"                       # bert, t5, gpt2 (ignored for graph)
TASKS="iso"                  # comma‐sep list (ignored for cod)
DATASET="webqsp"                   # e.g. webqsp, cwq, grailqa

# commented-out old paths:
# DATA_PATH="/data/user_data/zhenwu/data/webqsp/KBQA_walklet_shuffled_768_dim_data.dill"
# DATA_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_prediction/data/grailqa/KBQA_walklet_edge_label_1024_dim_data.dill"
# RELATIONS_PATH="/data/user_data/zhenwu/data/webqsp/relations_dict_shuffled.json"
# RELATIONS_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_prediction/data/grailqa/relations_dict.json"

# ─── pick the correct paths based on DATASET ──────────────────────────────────
if [[ "$DATASET" == "webqsp" ]]; then
  DATA_PATH="/data/user_data/zhenwu/data/webqsp/KBQA_walklet_shuffled_768_dim_data.dill"
  RELATIONS_PATH="/data/user_data/zhenwu/data/webqsp/relations_dict_shuffled.json"
  NODE_EMB_DIM=768
elif [[ "$DATASET" == "grailqa" ]]; then
  DATA_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_prediction/data/grailqa/KBQA_walklet_edge_label_1024_dim_data.dill"
  RELATIONS_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_prediction/data/grailqa/relations_dict.json"
  NODE_EMB_DIM=1024
else
  echo "Unknown dataset '$DATASET', please set DATA_PATH and RELATIONS_PATH manually."
  exit 1
fi

OUTPUT_DIR_BASE="/data/shire/projects/GenQA/ISOKBQA/isomorphism_prediction/ckpts"

PTLM_MODEL="t5-base"
BATCH_SIZE=8
EPOCHS=50
LR=5e-5
PATIENCE=5

SCHEDULER=none
STEP_SIZE=3
GAMMA=0.9
WARMUP_STEPS=0
T_MAX=${EPOCHS}
ETA_MIN=1e-5

INPUT_MAX_LENGTH=512
GENERATION_MAX_LENGTH=128
MAX_SEQ_LEN=512

GNN_MODEL="rgcn"
GNN_DEPTH=2
HIDDEN_DIM=768

SHARED_DIM=2048
DROPOUT=0.2

# ─── list of seeds to run sequentially ───────────────────────────────────────
# SEEDS=(18 42 15213 123456 987654)
SEEDS=(42) 

WEIGHTS="1"

WANDB_PROJECT="isomorphism_pred_KBQA"

# ─── loop over seeds ──────────────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
  CLEAN_TASKS="${TASKS//,/‐}"
  TS=$(date +%Y%m%d_%H%M%S)
  OUTPUT_DIR="${OUTPUT_DIR_BASE}/${DATASET}_${MODEL_TYPE}_${BACKBONE}_${CLEAN_TASKS}_seed${SEED}_${TS}"
  mkdir -p "$OUTPUT_DIR"

  WANDB_RUN="${DATASET}_${MODEL_TYPE}_${BACKBONE}_${CLEAN_TASKS}_seed${SEED}"

  echo "=== Seed $SEED → output: $OUTPUT_DIR, wandb run: $WANDB_RUN ==="

  python train.py \
    --model_type            "$MODEL_TYPE" \
    --backbone              "$BACKBONE" \
    --tasks                 "$TASKS" \
    --dataset               "$DATASET" \
    --data_path             "$DATA_PATH" \
    --relations_path        "$RELATIONS_PATH" \
    --output_dir            "$OUTPUT_DIR" \
    --ptlm_model            "$PTLM_MODEL" \
    --batch_size            "$BATCH_SIZE" \
    --epochs                "$EPOCHS" \
    --lr                    "$LR" \
    --patience              "$PATIENCE" \
    --scheduler             "$SCHEDULER" \
    --step_size             "$STEP_SIZE" \
    --gamma                 "$GAMMA" \
    --warmup_steps          "$WARMUP_STEPS" \
    --t_max                 "$T_MAX" \
    --eta_min               "$ETA_MIN" \
    --seed                  "$SEED" \
    --input_max_length      "$INPUT_MAX_LENGTH" \
    --generation_max_length "$GENERATION_MAX_LENGTH" \
    --max_seq_len           "$MAX_SEQ_LEN" \
    --gnn_model             "$GNN_MODEL" \
    --gnn_depth             "$GNN_DEPTH" \
    --node_emb_dim          "$NODE_EMB_DIM" \
    --hidden_dim            "$HIDDEN_DIM" \
    --shared_dim            "$SHARED_DIM" \
    --dropout               "$DROPOUT" \
    --wandb_proj            "$WANDB_PROJECT" \
    --wandb_run             "$WANDB_RUN" \
    ${WEIGHTS:+--weights "$WEIGHTS"}

  echo "=== Finished seed $SEED ==="
  echo
done