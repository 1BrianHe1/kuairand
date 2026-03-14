#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${RUN_NAME:-pure_cuda_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$BASE_DIR/checkpoints/runs/$RUN_NAME}"
PROCESSED_DIR="${PROCESSED_DIR:-$BASE_DIR/processed_pure}"

DEVICE="${DEVICE:-cuda}"
TOPK="${TOPK:-20,50}"
RECALL_TOPN="${RECALL_TOPN:-300}"
MAX_HISTORY_LEN="${MAX_HISTORY_LEN:-500}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"

RECALL_EPOCHS="${RECALL_EPOCHS:-10}"
RECALL_BATCH_SIZE="${RECALL_BATCH_SIZE:-512}"
RECALL_TRAIN_MAX_ROWS="${RECALL_TRAIN_MAX_ROWS:-}"
RECALL_VALID_MAX_ROWS="${RECALL_VALID_MAX_ROWS:-}"
RECALL_EVAL_EVERY="${RECALL_EVAL_EVERY:-2}"
RECALL_NUM_EXPLICIT_NEGATIVES="${RECALL_NUM_EXPLICIT_NEGATIVES:-0}"
RECALL_NUM_HARD_NEGATIVES="${RECALL_NUM_HARD_NEGATIVES:-0}"
RECALL_NUM_RANDOM_NEGATIVES="${RECALL_NUM_RANDOM_NEGATIVES:-24}"

RANK_EPOCHS="${RANK_EPOCHS:-10}"
RANK_BATCH_SIZE="${RANK_BATCH_SIZE:-512}"
RANK_TRAIN_MAX_ROWS="${RANK_TRAIN_MAX_ROWS:-}"
RANK_VALID_MAX_ROWS="${RANK_VALID_MAX_ROWS:-}"
RANK_EVAL_EVERY="${RANK_EVAL_EVERY:-2}"

TEST_MAX_ROWS="${TEST_MAX_ROWS:-}"
MAX_EVAL_USERS="${MAX_EVAL_USERS:-}"

RECALL_DIR="$RUN_DIR/recall"
RANK_DIR="$RUN_DIR/rank"
EVAL_DIR="$RUN_DIR/eval"
EVAL_JSON="$EVAL_DIR/test_metrics.json"

mkdir -p "$RECALL_DIR" "$RANK_DIR" "$EVAL_DIR"

echo "[pipeline] run_dir=$RUN_DIR"
echo "[pipeline] device=$DEVICE recall_topn=$RECALL_TOPN topk=$TOPK exact_topk=1"

recall_cmd=(
  python "$BASE_DIR/train_recall_twotower.py"
  --processed-dir "$PROCESSED_DIR"
  --output-dir "$RECALL_DIR"
  --device "$DEVICE"
  --amp
  --epochs "$RECALL_EPOCHS"
  --batch-size "$RECALL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --eval-every "$RECALL_EVAL_EVERY"
  --num-explicit-negatives "$RECALL_NUM_EXPLICIT_NEGATIVES"
  --num-hard-negatives "$RECALL_NUM_HARD_NEGATIVES"
  --num-random-negatives "$RECALL_NUM_RANDOM_NEGATIVES"
  --max-history-len "$MAX_HISTORY_LEN"
  --topk "$TOPK"
  --recall-topn "$RECALL_TOPN"
  --seed "$SEED"
)
if [[ -n "$RECALL_TRAIN_MAX_ROWS" ]]; then
  recall_cmd+=(--train-max-rows "$RECALL_TRAIN_MAX_ROWS")
fi
if [[ -n "$RECALL_VALID_MAX_ROWS" ]]; then
  recall_cmd+=(--valid-max-rows "$RECALL_VALID_MAX_ROWS")
fi
if [[ -n "$MAX_EVAL_USERS" ]]; then
  recall_cmd+=(--max-eval-users "$MAX_EVAL_USERS")
fi
"${recall_cmd[@]}"

rank_cmd=(
  python "$BASE_DIR/train_rank_shared_bottom.py"
  --processed-dir "$PROCESSED_DIR"
  --output-dir "$RANK_DIR"
  --device "$DEVICE"
  --amp
  --epochs "$RANK_EPOCHS"
  --batch-size "$RANK_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --eval-every "$RANK_EVAL_EVERY"
  --max-history-len "$MAX_HISTORY_LEN"
  --seed "$SEED"
)
if [[ -n "$RANK_TRAIN_MAX_ROWS" ]]; then
  rank_cmd+=(--train-max-rows "$RANK_TRAIN_MAX_ROWS")
fi
if [[ -n "$RANK_VALID_MAX_ROWS" ]]; then
  rank_cmd+=(--valid-max-rows "$RANK_VALID_MAX_ROWS")
fi
"${rank_cmd[@]}"

eval_cmd=(
  python "$BASE_DIR/evaluate_pipeline.py"
  --processed-dir "$PROCESSED_DIR"
  --recall-ckpt "$RECALL_DIR/recall_model.pt"
  --rank-ckpt "$RANK_DIR/rank_model.pt"
  --candidate-item-ids "$RECALL_DIR/candidate_item_ids.npy"
  --output-json "$EVAL_JSON"
  --device "$DEVICE"
  --topk "$TOPK"
  --recall-topn "$RECALL_TOPN"
  --max-history-len "$MAX_HISTORY_LEN"
  --seed "$SEED"
)
if [[ -n "$TEST_MAX_ROWS" ]]; then
  eval_cmd+=(--test-max-rows "$TEST_MAX_ROWS")
fi
if [[ -n "$MAX_EVAL_USERS" ]]; then
  eval_cmd+=(--max-eval-users "$MAX_EVAL_USERS")
fi
"${eval_cmd[@]}"

echo "[pipeline] finished"
echo "[pipeline] eval_json=$EVAL_JSON"
