#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESSED_DIR="${PROCESSED_DIR:-$BASE_DIR/processed_pure}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
TOPK="${TOPK:-50,100,200}"

MAIN_EPOCHS="${MAIN_EPOCHS:-20}"
MAIN_BATCH_SIZE="${MAIN_BATCH_SIZE:-512}"
MAIN_OUTPUT_DIR="${MAIN_OUTPUT_DIR:-$BASE_DIR/checkpoints/recall_pure_refresh_ep20}"

CONTENT_EPOCHS="${CONTENT_EPOCHS:-20}"
CONTENT_BATCH_SIZE="${CONTENT_BATCH_SIZE:-256}"
CONTENT_OUTPUT_DIR="${CONTENT_OUTPUT_DIR:-$BASE_DIR/checkpoints/content_recall_pure_refresh_ep20}"

HSTU_EPOCHS="${HSTU_EPOCHS:-20}"
HSTU_BATCH_SIZE="${HSTU_BATCH_SIZE:-128}"
HSTU_EVAL_BATCH_SIZE="${HSTU_EVAL_BATCH_SIZE:-256}"
HSTU_OUTPUT_DIR="${HSTU_OUTPUT_DIR:-$BASE_DIR/checkpoints/hstu_recall_pure_bias_nostatic_ep20}"
HSTU_DATA_DIR="${HSTU_DATA_DIR:-$PROCESSED_DIR/hstu_interleaved_firsttoken_len100}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-$BASE_DIR/checkpoints/recall_three_routes_pure_refresh}"
DIAGNOSTIC_JSON="${DIAGNOSTIC_JSON:-$ANALYSIS_ROOT/diagnostic_200_200_200.json}"
OVERLAP_JSON="${OVERLAP_JSON:-$ANALYSIS_ROOT/test_overlap.json}"
FUSION_SEARCH_JSON="${FUSION_SEARCH_JSON:-$ANALYSIS_ROOT/fusion_search_valid.json}"

BUDGET_TOTAL="${BUDGET_TOTAL:-200}"
BUDGET_STEP="${BUDGET_STEP:-20}"
MIN_MAIN_BUDGET="${MIN_MAIN_BUDGET:-20}"
MIN_CONTENT_BUDGET="${MIN_CONTENT_BUDGET:-20}"
MIN_HSTU_BUDGET="${MIN_HSTU_BUDGET:-20}"
ROUTE_TOPN="${ROUTE_TOPN:-200}"
RRF_K="${RRF_K:-60}"

RUN_FINAL_TEST="${RUN_FINAL_TEST:-0}"
FINAL_MAIN_TOPN="${FINAL_MAIN_TOPN:-20}"
FINAL_CONTENT_TOPN="${FINAL_CONTENT_TOPN:-20}"
FINAL_HSTU_TOPN="${FINAL_HSTU_TOPN:-160}"
FINAL_TEST_JSON="${FINAL_TEST_JSON:-$ANALYSIS_ROOT/final_test_${FINAL_HSTU_TOPN}_${FINAL_MAIN_TOPN}_${FINAL_CONTENT_TOPN}.json}"

mkdir -p "$MAIN_OUTPUT_DIR" "$CONTENT_OUTPUT_DIR" "$HSTU_OUTPUT_DIR" "$ANALYSIS_ROOT"

echo "[three-route] processed_dir=$PROCESSED_DIR"
echo "[three-route] device=$DEVICE topk=$TOPK seed=$SEED"
echo "[three-route] main_out=$MAIN_OUTPUT_DIR"
echo "[three-route] content_out=$CONTENT_OUTPUT_DIR"
echo "[three-route] hstu_out=$HSTU_OUTPUT_DIR"

main_cmd=(
  python "$BASE_DIR/train_recall_twotower.py"
  --processed-dir "$PROCESSED_DIR"
  --output-dir "$MAIN_OUTPUT_DIR"
  --epochs "$MAIN_EPOCHS"
  --batch-size "$MAIN_BATCH_SIZE"
  --topk "$TOPK"
  --device "$DEVICE"
  --amp
  --seed "$SEED"
)
"${main_cmd[@]}"

content_cmd=(
  python "$BASE_DIR/train_recall_content_twotower.py"
  --processed-dir "$PROCESSED_DIR"
  --output-dir "$CONTENT_OUTPUT_DIR"
  --epochs "$CONTENT_EPOCHS"
  --batch-size "$CONTENT_BATCH_SIZE"
  --topk "$TOPK"
  --device "$DEVICE"
  --amp
  --seed "$SEED"
)
"${content_cmd[@]}"

hstu_cmd=(
  python "$BASE_DIR/train_recall_hstu.py"
  --data-dir "$HSTU_DATA_DIR"
  --output-dir "$HSTU_OUTPUT_DIR"
  --epochs "$HSTU_EPOCHS"
  --batch-size "$HSTU_BATCH_SIZE"
  --eval-batch-size "$HSTU_EVAL_BATCH_SIZE"
  --topk "$TOPK"
  --disable-user-static-features
  --use-position-bias
  --use-time-bias
  --device "$DEVICE"
  --seed "$SEED"
)
"${hstu_cmd[@]}"

diagnostic_cmd=(
  python "$BASE_DIR/evaluate_recall_three_routes.py"
  --main-ckpt "$MAIN_OUTPUT_DIR/recall_model.pt"
  --main-candidate-item-ids "$MAIN_OUTPUT_DIR/candidate_item_ids.npy"
  --content-ckpt "$CONTENT_OUTPUT_DIR/content_recall_model.pt"
  --hstu-ckpt "$HSTU_OUTPUT_DIR/hstu_recall_model.pt"
  --hstu-data-dir "$HSTU_DATA_DIR"
  --main-topn "$ROUTE_TOPN"
  --content-topn "$ROUTE_TOPN"
  --hstu-topn "$ROUTE_TOPN"
  --fusion-method weighted_rrf
  --topk "$TOPK"
  --output-json "$DIAGNOSTIC_JSON"
  --device "$DEVICE"
  --seed "$SEED"
  --disable-progress
)
"${diagnostic_cmd[@]}"

overlap_cmd=(
  python "$BASE_DIR/compare_recall_hit_overlap.py"
  --main-ckpt "$MAIN_OUTPUT_DIR/recall_model.pt"
  --main-candidate-item-ids "$MAIN_OUTPUT_DIR/candidate_item_ids.npy"
  --content-ckpt "$CONTENT_OUTPUT_DIR/content_recall_model.pt"
  --hstu-ckpt "$HSTU_OUTPUT_DIR/hstu_recall_model.pt"
  --hstu-data-dir "$HSTU_DATA_DIR"
  --main-topn "$ROUTE_TOPN"
  --content-topn "$ROUTE_TOPN"
  --hstu-topn "$ROUTE_TOPN"
  --topk "$TOPK"
  --output-json "$OVERLAP_JSON"
  --device "$DEVICE"
  --seed "$SEED"
  --disable-progress
)
"${overlap_cmd[@]}"

fusion_search_cmd=(
  python "$BASE_DIR/search_multi_route_fusion.py"
  --split valid
  --main-ckpt "$MAIN_OUTPUT_DIR/recall_model.pt"
  --main-candidate-item-ids "$MAIN_OUTPUT_DIR/candidate_item_ids.npy"
  --content-ckpt "$CONTENT_OUTPUT_DIR/content_recall_model.pt"
  --hstu-ckpt "$HSTU_OUTPUT_DIR/hstu_recall_model.pt"
  --hstu-data-dir "$HSTU_DATA_DIR"
  --route-topn "$ROUTE_TOPN"
  --budget-total "$BUDGET_TOTAL"
  --budget-step "$BUDGET_STEP"
  --min-main-budget "$MIN_MAIN_BUDGET"
  --min-content-budget "$MIN_CONTENT_BUDGET"
  --min-hstu-budget "$MIN_HSTU_BUDGET"
  --methods weighted_rrf
  --rrf-k "$RRF_K"
  --topk "$TOPK"
  --output-json "$FUSION_SEARCH_JSON"
  --device "$DEVICE"
  --seed "$SEED"
  --disable-progress
)
"${fusion_search_cmd[@]}"

if [[ "$RUN_FINAL_TEST" == "1" ]]; then
  final_test_cmd=(
    python "$BASE_DIR/evaluate_recall_three_routes.py"
    --main-ckpt "$MAIN_OUTPUT_DIR/recall_model.pt"
    --main-candidate-item-ids "$MAIN_OUTPUT_DIR/candidate_item_ids.npy"
    --content-ckpt "$CONTENT_OUTPUT_DIR/content_recall_model.pt"
    --hstu-ckpt "$HSTU_OUTPUT_DIR/hstu_recall_model.pt"
    --hstu-data-dir "$HSTU_DATA_DIR"
    --main-topn "$FINAL_MAIN_TOPN"
    --content-topn "$FINAL_CONTENT_TOPN"
    --hstu-topn "$FINAL_HSTU_TOPN"
    --fusion-method weighted_rrf
    --topk "$TOPK"
    --output-json "$FINAL_TEST_JSON"
    --device "$DEVICE"
    --seed "$SEED"
    --disable-progress
  )
  "${final_test_cmd[@]}"
fi

echo "[three-route] done"
echo "[three-route] diagnostic_json=$DIAGNOSTIC_JSON"
echo "[three-route] overlap_json=$OVERLAP_JSON"
echo "[three-route] fusion_search_json=$FUSION_SEARCH_JSON"
if [[ "$RUN_FINAL_TEST" == "1" ]]; then
  echo "[three-route] final_test_json=$FINAL_TEST_JSON"
else
  echo "[three-route] final test skipped; set RUN_FINAL_TEST=1 after choosing budgets from valid search"
fi
