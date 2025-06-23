#!/usr/bin/env bash
#SBATCH --job-name=ISO_KBQA
#SBATCH --partition=shire-general
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --output=/home/zhenwu/isomorphism_pred_KBQA/logs/%x_%j.out
#SBATCH --mem=80GB
#SBATCH --time=2-00:00:00

set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
MODEL_TYPE="text"                   # text, graph, or cod
BACKBONE="t5"                       # bert, t5, gpt2 (ignored for graph)
TASKS="ranking"      # comma‐sep list (ignored for cod)
DATASET="webqsp"                    # e.g. webqsp, cwq, grailqa
# DATA_PATH="/data/user_data/zhenwu/data/webqsp/KBQA_walklet_shuffled_768_dim_data.dill"
DATA_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_prediction/data/grailqa/KBQA_walklet_edge_label_1024_dim_data.dill"
OUTPUT_DIR=""                       # if blank, auto‐named below

PTLM_MODEL="t5-base"
BATCH_SIZE=16
EPOCHS=80
LR=8e-5
PATIENCE=8

SCHEDULER=none
STEP_SIZE=3
GAMMA=0.9
WARMUP_STEPS=0
T_MAX=${EPOCHS}  # for cosine scheduler
ETA_MIN=1e-5

INPUT_MAX_LENGTH=512
GENERATION_MAX_LENGTH=128
MAX_SEQ_LEN=512

GNN_MODEL="rgcn"        # rgcn, rgat, gcn
GNN_DEPTH=2
NODE_EMB_DIM=768

SHARED_DIM=2048         # for CoD
DROPOUT=0.2

SEED=15213

WEIGHTS="1"              # e.g. "0.7,0.2,0.1"
# RELATIONS_PATH="/data/user_data/zhenwu/data/webqsp/relations_dict_shuffled.json"  
RELATIONS_PATH="/data/shire/projects/GenQA/ISOKBQA/isomorphism_prediction/data/grailqa/relations_dict.json"

WANDB_PROJECT="isomorphism_pred_KBQA"
WANDB_RUN="${DATASET}_${MODEL_TYPE}_${BACKBONE}_${TASKS//,/‐}"

# ─── Parse command‐line flags ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_type)           MODEL_TYPE="$2"; shift 2;;
    --backbone)             BACKBONE="$2"; shift 2;;
    --tasks)                TASKS="$2"; shift 2;;
    --dataset)              DATASET="$2"; shift 2;;
    --data_path)            DATA_PATH="$2"; shift 2;;
    --output_dir)           OUTPUT_DIR="$2"; shift 2;;

    --ptlm_model)           PTLM_MODEL="$2"; shift 2;;
    --batch_size)           BATCH_SIZE="$2"; shift 2;;
    --epochs)               EPOCHS="$2"; shift 2;;
    --lr)                   LR="$2"; shift 2;;

    --patience)             PATIENCE="$2"; shift 2;;
    --scheduler)            SCHEDULER="$2"; shift 2;;
    --step_size)            STEP_SIZE="$2"; shift 2;;
    --gamma)                GAMMA="$2"; shift 2;;
    --warmup_steps)         WARMUP_STEPS="$2"; shift 2;;
    --t_max)                T_MAX="$2"; shift 2;;
    --eta_min)              ETA_MIN="$2"; shift 2;;
    --seed)                 SEED="$2"; shift 2;;

    --input_max_length)     INPUT_MAX_LENGTH="$2"; shift 2;;
    --generation_max_length)GENERATION_MAX_LENGTH="$2"; shift 2;;
    --max_seq_len)          MAX_SEQ_LEN="$2"; shift 2;;

    --gnn_model)            GNN_MODEL="$2"; shift 2;;
    --gnn_depth)            GNN_DEPTH="$2"; shift 2;;
    --node_emb_dim)         NODE_EMB_DIM="$2"; shift 2;;

    --shared_dim)           SHARED_DIM="$2"; shift 2;;
    --dropout)              DROPOUT="$2"; shift 2;;

    --weights)              WEIGHTS="$2"; shift 2;;
    --relations_path)       RELATIONS_PATH="$2"; shift 2;;

    --wandb_proj)           WANDB_PROJECT="$2"; shift 2;;
    --wandb_run)            WANDB_RUN="$2"; shift 2;;

    *) echo "Unknown flag: $1"; exit 1;;
  esac
done

# ─── Auto‐name output if not given ─────────────────────────────────────────────
if [[ -z "$OUTPUT_DIR" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  CLEAN_TASKS="${TASKS//,/‐}"
  OUTPUT_DIR="/data/shire/projects/GenQA/ISOKBQA/isomorphism_prediction/ckpts/${DATASET}_${MODEL_TYPE}_${BACKBONE}_${CLEAN_TASKS}_${TS}"
fi
mkdir -p "$OUTPUT_DIR"

# ─── Echo configuration ───────────────────────────────────────────────────────
echo "Launching training with:"
cat <<EOF
  MODEL_TYPE            = $MODEL_TYPE
  BACKBONE              = $BACKBONE
  TASKS                 = $TASKS
  DATASET               = $DATASET
  DATA_PATH             = $DATA_PATH
  OUTPUT_DIR            = $OUTPUT_DIR

  PTLM_MODEL            = $PTLM_MODEL
  BATCH_SIZE            = $BATCH_SIZE
  EPOCHS                = $EPOCHS
  LR                    = $LR
  PATIENCE              = $PATIENCE

  SCHEDULER             = $SCHEDULER
  STEP_SIZE             = $STEP_SIZE
  GAMMA                 = $GAMMA
  WARMUP_STEPS          = $WARMUP_STEPS
  T_MAX                 = $T_MAX
  ETA_MIN               = $ETA_MIN
  SEED                  = $SEED

  INPUT_MAX_LENGTH      = $INPUT_MAX_LENGTH
  GENERATION_MAX_LENGTH = $GENERATION_MAX_LENGTH
  MAX_SEQ_LEN           = $MAX_SEQ_LEN

  GNN_MODEL             = $GNN_MODEL
  GNN_DEPTH             = $GNN_DEPTH
  NODE_EMB_DIM          = $NODE_EMB_DIM

  SHARED_DIM            = $SHARED_DIM
  DROPOUT               = $DROPOUT

  WEIGHTS               = $WEIGHTS
  RELATIONS_PATH        = $RELATIONS_PATH

  WANDB_PROJECT         = $WANDB_PROJECT
  WANDB_RUN             = $WANDB_RUN
EOF
echo

# ─── Run python ────────────────────────────────────────────────────────────────
python train.py \
  --model_type            "$MODEL_TYPE" \
  --backbone              "$BACKBONE" \
  --tasks                 "$TASKS" \
  --dataset               "$DATASET" \
  --data_path             "$DATA_PATH" \
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
  --shared_dim            "$SHARED_DIM" \
  --dropout               "$DROPOUT" \
  --wandb_proj            "$WANDB_PROJECT" \
  --wandb_run             "$WANDB_RUN" \
  ${WEIGHTS:+--weights "$WEIGHTS"} \
  ${RELATIONS_PATH:+--relations_path "$RELATIONS_PATH"}