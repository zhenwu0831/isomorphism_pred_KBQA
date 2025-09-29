#!/usr/bin/env bash

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export CUDA_LAUNCH_BLOCKING=1

# ─── Defaults ──────────────────────────────────────────────────────────────────
MODEL_TYPE="text"                   # text, graph, or cod
BACKBONE="t5"                       # bert, t5, gpt2 (ignored for graph)
TASKS="generation"                  # comma‐sep list (ignored for cod)
DATASET="grailqa"                   # e.g. webqsp, cwq, grailqa

# ─── pick the correct paths based on DATASET ──────────────────────────────────
DATA_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/${DATASET}/KBQA_walklet_edge_label_768_dim_data.dill"
RELATIONS_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/${DATASET}/relations_dict.json"

OUTPUT_DIR_BASE="/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/ckpts_rdutt"

PTLM_MODEL="t5-base"  # bert-base-uncased, t5-base, gpt2 (ignored for graph)
BATCH_SIZE=4
EPOCHS=50
LR=5e-5
PATIENCE=5

SCHEDULER=none
STEP_SIZE=3
GAMMA=0.9
WARMUP_STEPS=0
T_MAX=${EPOCHS}
ETA_MIN=1e-5

INPUT_MAX_LENGTH=1024
GENERATION_MAX_LENGTH=128
MAX_SEQ_LEN=512

GNN_MODEL="gat"  # 
GNN_DEPTH=2
HIDDEN_DIM=128
NODE_EMB_DIM=768


SHARED_DIM=2048
DROPOUT=0.2

# ─── list of seeds to run sequentially ───────────────────────────────────────
# SEEDS=(18 42 15213 123456 987654)
# SEEDS=(123456 987654)
SEEDS=(123456 987654) 

WEIGHTS="1"

WANDB_PROJECT="isomorphism_pred_KBQA"

# ─── loop over seeds ──────────────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
  CLEAN_TASKS="${TASKS//,/‐}"
  TS=$(date +%Y%m%d_%H%M%S)

  if [[ "$MODEL_TYPE" == "graph" ]]; then
    # include GNN_MODEL in names
    NAME="${DATASET}_${MODEL_TYPE}_${GNN_MODEL}_${GNN_DEPTH}_${HIDDEN_DIM}_${CLEAN_TASKS}_seed${SEED}_${TS}"
    RUN_NAME="${DATASET}_${MODEL_TYPE}_${GNN_MODEL}_${GNN_DEPTH}_${HIDDEN_DIM}_${CLEAN_TASKS}_seed${SEED}"
  elif [[ "$MODEL_TYPE" == "cod" ]]; then
    # include BACKBONE in names
    NAME="${DATASET}_${MODEL_TYPE}_${BACKBONE}_${GNN_MODEL}_${GNN_DEPTH}_${HIDDEN_DIM}_${CLEAN_TASKS}_seed${SEED}_${TS}"
    RUN_NAME="${DATASET}_${MODEL_TYPE}_${BACKBONE}_${GNN_MODEL}_${GNN_DEPTH}_${HIDDEN_DIM}_${CLEAN_TASKS}_seed${SEED}"
  else
    # just use DATASET, MODEL_TYPE, and TASKS
    NAME="${DATASET}_${MODEL_TYPE}_${BACKBONE}_${CLEAN_TASKS}_seed${SEED}_${TS}"
    RUN_NAME="${DATASET}_${MODEL_TYPE}_${BACKBONE}_${CLEAN_TASKS}_seed${SEED}"
  fi

  OUTPUT_DIR="${OUTPUT_DIR_BASE}/${NAME}"
  mkdir -p "$OUTPUT_DIR"

  WANDB_RUN="$RUN_NAME"

  echo "=== Seed $SEED → output: $OUTPUT_DIR, wandb run: $WANDB_RUN ==="

  CUDA_VISIBLE_DEVICES=0  python train.py \
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
