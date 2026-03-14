# KuaiRand Baseline Data Pipeline

This folder contains the baseline data builder for:
- recall: single-route two-tower
- rank: shared-bottom multi-task (`is_click`, `long_view`)

The pipeline now supports both raw dataset variants:
- `Pure` via `--dataset-version pure` or by pointing `--data-dir` to `KuaiRand-Pure/data`
- `1K` via `--dataset-version 1k` or by pointing `--data-dir` to `KuaiRand-1K/data`

Default settings now point to `Pure`:
- raw data: `KuaiRand-Pure/data`
- processed data: `baseline/processed_pure`
- checkpoints: `checkpoints/*_pure`

If `--dataset-version auto` is used, `dataset.py` infers the variant from the filenames under `--data-dir`.

## Fixed split

- warmup: `2022-04-08` to `2022-04-21`
- train: `2022-04-22` to `2022-05-01`
- valid: `2022-05-02` to `2022-05-05`
- test: `2022-05-06` to `2022-05-08`

`warmup` is used only to initialize user history; it is not exported as supervised samples.

## Feature policy

- Uses only the matching standard logs / selected side tables for the chosen raw variant:
  - `log_standard_4_08_to_4_21_{pure|1k}.csv`
  - `log_standard_4_22_to_5_08_{pure|1k}.csv`
  - `user_features_{pure|1k}.csv`
  - `video_features_basic_{pure|1k}.csv`
- Excludes:
  - `log_random_4_22_to_5_08_{pure|1k}.csv`
  - `video_features_statistic_{pure|1k}.csv`

Each output sample includes:
- context: `user_id`, `video_id`, `date`, `hourmin`, `tab`, `time_ms`
- labels: `is_click`, `long_view`
- derived context: `hour_bucket`, `date_bucket`
- sequence: `history_pos_video_ids`, `history_pos_len`

Positive history is defined as `is_click == 1 OR long_view == 1`.

Current recall setup keeps only the single dual-tower retrieval path.
The earlier author/tag/fresh/hot multi-route merge has been removed.

## Run

```bash
python /home/hfx/KuaiRand/baseline/dataset.py \
  --data-dir /home/hfx/KuaiRand/KuaiRand-Pure/data \
  --dataset-version pure \
  --output-dir /home/hfx/KuaiRand/baseline/processed_pure \
  --max-history-len 500
```

Debug run (small sample):

```bash
python /home/hfx/KuaiRand/baseline/dataset.py \
  --nrows-warmup 200000 \
  --nrows-main 300000 \
  --output-dir /home/hfx/KuaiRand/baseline/processed_debug
```

## Train recall (two-tower + in-batch negative)

```bash
python /home/hfx/KuaiRand/baseline/train_recall_twotower.py \
  --processed-dir /home/hfx/KuaiRand/baseline/processed_pure \
  --output-dir /home/hfx/KuaiRand/baseline/checkpoints/recall_pure \
  --epochs 2 \
  --batch-size 512 \
  --train-max-rows 1500000 \
  --label-mode click_or_long \
  --topk 20,50
```

Validation now uses all positive exposures inside the `valid` split:
- user features: positive history before each valid exposure
- targets: the positive item for that exposure point
- sequence behavior: history grows within the `valid` split as time advances
- metrics: `hr@K`, `ndcg@K`

Candidate retrieval uses exact full-corpus top-k search over the recall candidate pool.

Main outputs:
- `checkpoints/recall_pure/recall_model.pt`
- `checkpoints/recall_pure/candidate_item_ids.npy`
- `checkpoints/recall_pure/recall_train_summary.json`

## Train ranker (shared-bottom, click + long_view)

```bash
python /home/hfx/KuaiRand/baseline/train_rank_shared_bottom.py \
  --processed-dir /home/hfx/KuaiRand/baseline/processed_pure \
  --output-dir /home/hfx/KuaiRand/baseline/checkpoints/rank_pure \
  --epochs 2 \
  --batch-size 512 \
  --train-max-rows 2000000
```

Main outputs:
- `checkpoints/rank_pure/rank_model.pt`
- `checkpoints/rank_pure/rank_train_summary.json`

## End-to-end evaluation

```bash
python /home/hfx/KuaiRand/baseline/evaluate_pipeline.py \
  --processed-dir /home/hfx/KuaiRand/baseline/processed_pure \
  --recall-ckpt /home/hfx/KuaiRand/baseline/checkpoints/recall_pure/recall_model.pt \
  --rank-ckpt /home/hfx/KuaiRand/baseline/checkpoints/rank_pure/rank_model.pt \
  --candidate-item-ids /home/hfx/KuaiRand/baseline/checkpoints/recall_pure/candidate_item_ids.npy \
  --output-json /home/hfx/KuaiRand/baseline/checkpoints/eval_pure/test_metrics.json \
  --topk 20,50 \
  --max-eval-users 500
```

The evaluation script reports:
- recall stage: `recall_hr@K`, `recall_ndcg@K`
- end-to-end rerank: `e2e_hr@K`, `e2e_ndcg@K`
- ranker pointwise: `auc_click`, `auc_long_view`

## One-click run

Run all stages in one command:

```bash
python /home/hfx/KuaiRand/baseline/run_all.py \
  --processed-dir /home/hfx/KuaiRand/baseline/processed_pure \
  --device cpu
```

Quick smoke run:

```bash
python /home/hfx/KuaiRand/baseline/run_all.py \
  --processed-dir /home/hfx/KuaiRand/baseline/processed_pure \
  --device cpu \
  --smoke
```

Rebuild data + run all:

```bash
python /home/hfx/KuaiRand/baseline/run_all.py \
  --rebuild-data \
  --data-dir /home/hfx/KuaiRand/KuaiRand-Pure/data \
  --dataset-version pure \
  --processed-dir /home/hfx/KuaiRand/baseline/processed_pure \
  --device cpu
```

To switch to `1K`, replace `--data-dir` and set `--dataset-version 1k`.
