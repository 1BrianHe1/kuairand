# KuaiRand Baseline Iteration Log

This file records major baseline runs so later iterations can compare:
- data version
- model structure
- training trajectory
- final evaluation metrics
- notable issues or follow-up hypotheses

## Run 20260311_222413

### Scope

- Data version: `KuaiRand-Pure`
- Processed data: `/mnt/sda/hfx/KuaiRand/baseline/processed_pure`
- Run dir: `/mnt/sda/hfx/KuaiRand/baseline/checkpoints/runs/20260311_222413`
- Seed: `42`
- Device: `cuda`
- Recall epochs: `10`
- Rank epochs: `10`
- Shared history length: `500`

### Data snapshot

- Users: `27285`
- Items: `7583`
- Train rows: `182742`
- Valid rows: `69487`
- Test rows: `43268`
- Train click rate: `0.4461`
- Train long-view rate: `0.3140`
- Valid click rate: `0.4455`
- Valid long-view rate: `0.3142`
- Test click rate: `0.4396`
- Test long-view rate: `0.3094`

### Current model structure

#### Recall model

- Model class: `TwoTowerRecallModel`
- Source: `/mnt/sda/hfx/KuaiRand/baseline/models.py`
- User tower inputs:
  - `23` user categorical features -> embedding
  - `3` context categorical features -> embedding
  - `4` user numeric features -> linear projection
  - positive history -> mean pooling over `video_id` embedding
- Item tower inputs:
  - `7` item categorical features -> embedding
  - `3` item numeric features -> linear projection
- Architecture:
  - `embedding_dim = 16`
  - user MLP: `[(23 + 3 + 1 + 1) * 16 = 448] -> 128 -> 64`
  - item MLP: `[(7 + 1) * 16 = 128] -> 128 -> 64`
  - user/item outputs are `L2` normalized before retrieval
- Training setup:
  - label mode: `click_or_long`
  - loss: query-level sampled softmax
  - negatives: `24` random, `0` explicit, `0` hard
  - retrieval topn before rerank: `300`
  - multi-route recall:
    - main `180`
    - author `80`
    - tag `80`
    - fresh `40`
    - hot `30`

#### Rank model

- Model class: `SharedBottomRanker`
- Source: `/mnt/sda/hfx/KuaiRand/baseline/models.py`
- Inputs:
  - same user/item/context features as baseline pipeline
  - positive history pooled from `video_id` embedding
- Architecture:
  - `embedding_dim = 16`
  - user representation: `23 * 16 + 16 + 16 = 400`
  - item representation: `7 * 16 + 16 = 128`
  - context representation: `3 * 16 = 48`
  - explicit cross term: `1`
  - shared bottom: `577 -> 128 -> 128`
  - click head: `128 -> 64 -> 1`
  - long-view head: `128 -> 64 -> 1`
- Training setup:
  - tasks: `click`, `long_view`
  - loss: weighted BCE
  - click loss weight: `1.0`
  - long-view loss weight: `1.0`

### Commands used

```bash
python /home/hfx/KuaiRand/baseline/run_all.py \
  --rebuild-data \
  --device cuda \
  --num-workers 4 \
  --recall-epochs 10 \
  --rank-epochs 10
```

### Recall training trajectory

- Checkpoint selection metric: `valid hr@50`
- Validation protocol:
  - pointwise recall evaluation over all positive exposures in `valid`
  - each eval point uses the interaction history available before that exposure
  - the within-split user history grows over time
- Best checkpoint epoch: `7`
- Best valid metrics:
  - `hr@20 = 0.06742`
  - `hr@50 = 0.12616`
  - `ndcg@20 = 0.04247`
  - `ndcg@50 = 0.06074`
  - `num_eval_points = 31016`
  - `num_unique_users = 13188`

| epoch | valid hr@50 | valid ndcg@50 |
| --- | ---: | ---: |
| 1 | 0.07822 | 0.04132 |
| 2 | 0.10656 | 0.05310 |
| 3 | 0.10814 | 0.05114 |
| 4 | 0.11991 | 0.05933 |
| 5 | 0.12242 | 0.05889 |
| 6 | 0.12326 | 0.06044 |
| 7 | 0.12616 | 0.06074 |
| 8 | 0.12606 | 0.06133 |
| 9 | 0.12352 | 0.05993 |
| 10 | 0.12161 | 0.05919 |

Observation:
- Under the new pointwise validation protocol, recall improved steadily through epoch `7`.
- `hr@50` peaked at epoch `7`, while `ndcg@50` was slightly higher at epoch `8`.
- After epoch `7-8`, both metrics flattened or softened, so the earlier early-stop hypothesis still holds.

### Rank training trajectory

- Checkpoint selection metric: average of `auc_click` and `auc_long_view`
- Best checkpoint epoch: `5`
- Best valid metrics:
  - `auc_click = 0.72206`
  - `auc_long_view = 0.71890`
  - `valid loss = 1.17489`

| epoch | valid auc_click | valid auc_long_view |
| --- | ---: | ---: |
| 1 | 0.68824 | 0.68550 |
| 2 | 0.70697 | 0.70450 |
| 3 | 0.71700 | 0.71400 |
| 4 | 0.72129 | 0.71874 |
| 5 | 0.72206 | 0.71890 |
| 6 | 0.71916 | 0.71679 |
| 7 | 0.72032 | 0.71856 |
| 8 | 0.71987 | 0.71782 |
| 9 | 0.71817 | 0.71510 |
| 10 | 0.71712 | 0.71562 |

Observation:
- Rank model improved steadily until epoch `5`.
- After epoch `5`, both task AUCs flattened or declined slightly.
- The current ranker still looks best with earlier stopping than `10` epochs.

### Final test metrics

#### Recall and end-to-end

- Evaluation protocol:
  - pointwise recall evaluation over all positive exposures in `test`
  - each test point uses the interaction history available before that exposure
  - the within-split user history grows over time
- `recall_hr@20 = 0.06034`
- `recall_hr@50 = 0.11456`
- `recall_ndcg@20 = 0.03809`
- `recall_ndcg@50 = 0.05498`
- `e2e_hr@20 = 0.02619`
- `e2e_hr@50 = 0.06469`
- `e2e_ndcg@20 = 0.01419`
- `e2e_ndcg@50 = 0.02610`

#### Ranker pointwise AUC on test exposures

- `auc_click = 0.72213`
- `auc_long_view = 0.71685`
- `num_rows = 43268`

#### Candidate routing stats

- `num_eval_points = 19091`
- `num_unique_users = 10130`
- `num_candidates = 7583`
- `avg_merged_candidates = 285.05`
- `avg_selected_main = 180.00`
- `avg_selected_author = 7.24`
- `avg_selected_tag = 67.39`
- `avg_selected_fresh = 18.40`
- `avg_selected_hot = 12.02`

### Raw eval snapshot

```json
{
  "recall_rerank_metrics": {
    "recall_hr@20": 0.06034256979728668,
    "recall_hr@50": 0.11455659734953644,
    "recall_ndcg@20": 0.038091163975986074,
    "recall_ndcg@50": 0.05497752869109249,
    "e2e_hr@20": 0.026190351474516788,
    "e2e_hr@50": 0.06469016814205647,
    "e2e_ndcg@20": 0.014188370121556118,
    "e2e_ndcg@50": 0.026096554612657242,
    "num_users": 19091.0,
    "num_eval_points": 19091.0,
    "num_unique_users": 10130.0,
    "num_skipped": 0.0,
    "num_candidates": 7583.0,
    "avg_merged_candidates": 285.0492378607721,
    "avg_selected_main": 180.0,
    "avg_pool_main": 180.0,
    "avg_selected_author": 7.237808391388612,
    "avg_pool_author": 7.896705253784506,
    "avg_selected_tag": 67.39212194227646,
    "avg_pool_tag": 79.5039023623697,
    "avg_selected_fresh": 18.395526687968154,
    "avg_pool_fresh": 40.0,
    "avg_selected_hot": 12.023780839138862,
    "avg_pool_hot": 29.989419098004294
  },
  "ranking_auc_metrics": {
    "auc_click": 0.722133126733011,
    "auc_long_view": 0.7168537824552258,
    "num_rows": 43268.0
  }
}
```

### Notes for iteration history

- This run is the first logged run under the new pointwise recall validation and test protocol over all positive exposures, so its recall metrics should not be compared directly with older per-user-first-step results.
- `num_eval_points` and `num_unique_users` are now logged explicitly because the evaluation unit is no longer one point per user.
- This run requested `faiss hnsw`, but logs still showed `faiss index.add` failed and retrieval fell back to exact torch top-k search.
- Therefore, the quality metrics above still reflect the exact-search fallback path rather than a true HNSW retrieval path.
- The evaluation script had already been patched so ranking AUC evaluation can run under `cuda + num_workers > 0`.

### Next iteration candidates

- Try early stopping around recall epoch `7` and rank epoch `5`.
- Investigate the Faiss `input not a numpy array` warning so HNSW can be measured correctly.
- The gap from `recall_hr@50 = 0.11456` to `e2e_hr@50 = 0.06469` suggests rerank-stage scoring or candidate composition is still a bottleneck.
- Priority tuning directions:
  - score weights for `click` vs `long_view`
  - stronger hard negatives for recall
  - more expressive cross features in the ranker
  - route allocation across `author/tag/fresh/hot`

## Recall Route Compare 20260312

### Scope

- Data version: `KuaiRand-Pure`
- Processed data: `/mnt/sda/hfx/KuaiRand/baseline/processed_pure`
- Evaluation type: recall-only comparison, no ranking model involved
- Evaluation protocol:
  - pointwise recall evaluation over all positive exposures in `valid` and `test`
  - target label mode: `click_or_long`
  - candidate cutoff per route: `200`
  - metrics: `HR/Recall@50,100,200` and `NDCG@50,100,200`
- Assets / checkpoints:
  - main dual-tower: `/mnt/sda/hfx/KuaiRand/baseline/checkpoints/recall_pure/recall_model.pt`
  - content route assets: `/mnt/sda/hfx/KuaiRand/baseline/processed_pure/content_assets`
  - HSTU route: `/mnt/sda/hfx/KuaiRand/baseline/checkpoints/hstu_recall_pure/hstu_recall_model.pt`
  - comparison result: `/mnt/sda/hfx/KuaiRand/baseline/checkpoints/recall_only_compare/metrics.json`

### Current Recall Routes

#### Main route: learned dual-tower recall

- Model: `TwoTowerRecallModel`
- Features:
  - `23` user categorical fields
  - `3` context categorical fields
  - `4` user numeric fields
  - positive interaction history pooled from `video_id` embedding
  - `7` item categorical fields
  - `3` item numeric fields
- Modeling:
  - learned user tower + learned item tower
  - query-level sampled softmax training
  - retrieval by user/item inner product

#### Content route: zero-shot content recall, no learning

- This route does not train a recall model.
- Item-side features used to build unified text:
  - `caption`
  - `first_level_category_name`
  - `second_level_category_name`
  - `tag`
  - `video_type`
  - `author_id`
- Text construction:
  - template prefix is `passage:`
  - missing fields are kept empty
  - all fields are converted to string
  - caption is truncated to at most `256` characters
- Item modeling:
  - text is encoded offline by `intfloat/multilingual-e5-base`
  - output is L2-normalized content embedding
- User modeling:
  - use recent `5` high-quality history interactions
  - strong positive: any of `long_view / is_like / is_follow / is_comment / is_forward`
  - weak positive: `is_click == 1` and `is_hate == 0`, but not strong positive
  - strong / weak weights: `1.0 / 0.5`
  - time decay half-life: `48h`
  - final user content vector is the weighted average of matched history item content embeddings
- Retrieval:
  - top-k by dot product between user content vector and offline item content embedding bank

#### HSTU route: learned sequential recall

- Data builder: `build_hstu_kuairand_data.py`
- Model / trainer: `train_recall_hstu.py`
- Sequence features:
  - positive-only interaction sequence
  - `item_id`
  - `signal_id`
  - `timestamp`
- Signal definition:
  - `4`: `is_follow / is_comment / is_forward`
  - `3`: `is_like`
  - `2`: `long_view`
  - `1`: `is_click == 1` and `is_hate == 0`
- Modeling:
  - HSTU sequential encoder
  - token layout: `interleaved`
  - user feature mode: `first_token`
  - max sequence user interaction history length: `100`
  - next-item retrieval over full candidate pool

### Recall-Only Comparison

#### Valid

| route | hr@50 | hr@100 | hr@200 | ndcg@50 | ndcg@100 | ndcg@200 | avg_candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 0.1260 | 0.1979 | 0.2991 | 0.0613 | 0.0797 | 0.1021 | 200.00 |
| content | 0.0200 | 0.0349 | 0.0596 | 0.0089 | 0.0128 | 0.0182 | 198.54 |
| hstu | 0.2323 | 0.3645 | 0.5359 | 0.1051 | 0.1389 | 0.1769 | 198.51 |
| merged | 0.1260 | 0.1979 | 0.2991 | 0.0613 | 0.0797 | 0.1021 | 527.94 |

- Valid eval points: `31016`
- Valid unique users: `13188`
- HSTU matched queries: `30785`

#### Test

| route | hr@50 | hr@100 | hr@200 | ndcg@50 | ndcg@100 | ndcg@200 | avg_candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 0.1149 | 0.1808 | 0.2720 | 0.0555 | 0.0724 | 0.0925 | 200.00 |
| content | 0.0214 | 0.0356 | 0.0599 | 0.0098 | 0.0134 | 0.0188 | 198.93 |
| hstu | 0.2066 | 0.3288 | 0.4867 | 0.0948 | 0.1261 | 0.1610 | 198.91 |
| merged | 0.1149 | 0.1808 | 0.2720 | 0.0555 | 0.0724 | 0.0925 | 530.61 |

- Test eval points: `19091`
- Test unique users: `10130`
- HSTU matched queries: `18987`

### Interpretation

- The current standalone ranking is `HSTU > main dual-tower >> zero-shot content`.
- The content route is much weaker because it is not learned from KuaiRand interaction supervision; it is only a text-embedding similarity retrieval route.
- Under the current merge logic, `merged` is effectively identical to `main` at `@50/@100/@200`.
- Reason:
  - current merge keeps the main route's top `200` first
  - content and HSTU candidates are appended after deduplication
  - therefore the extra routes do not enter the evaluation cutoff for `K <= 200`

### Notes

- This comparison is recall-only. It should not be mixed with earlier end-to-end rerank metrics.
- In the current pointwise single-target evaluation setup, `Recall@K` and `HR@K` are numerically the same.
- The content route recorded here is the no-learning version, not the later `content two-tower` training script.

## Learned Content Two-Tower 20260312

### Scope

- Data version: `KuaiRand-Pure`
- Processed data: `/home/hfx/KuaiRand/baseline/processed_pure`
- Content recall checkpoint: `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_pure_v2/content_recall_model.pt`
- Training summary: `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_pure_v2/content_recall_train_summary.json`
- Test-time route overlap result: `/home/hfx/KuaiRand/baseline/checkpoints/recall_route_overlap/test_overlap.json`
- Fixed-budget fusion search result: `/home/hfx/KuaiRand/baseline/checkpoints/recall_route_overlap/fusion_search.json`

### Model Structure

#### Content two-tower route

- Model: `ContentTwoTowerRecallModel`
- Source:
  - `/home/hfx/KuaiRand/baseline/models.py`
  - `/home/hfx/KuaiRand/baseline/train_recall_content_twotower.py`
- Item tower:
  - input is offline content embedding from `intfloat/multilingual-e5-base`
  - item text fields:
    - `caption`
    - `category_l1`
    - `category_l2`
    - `tag`
    - `video_type`
    - `author_id`
  - item text uses `passage:` prefix
  - missing fields are kept empty
  - caption is truncated to `256` characters
  - item tower is a trainable MLP projection on top of fixed `768`-dim content embeddings
- User tower:
  - no `user_id`
  - no heavy static user profile features
  - lightweight context features only:
    - `tab`
    - `hour_bucket`
    - `weekday`
    - `is_weekend`
    - `short_history_len` bucket
    - `short_strong_count` bucket
    - `long_strong_history_len` bucket
  - short-term content branch:
    - use recent `15` high-quality interactions from `content_history_*`
    - history content embeddings are weighted by `behavior_weight x time_decay`
    - signal weights:
      - `click = 0.5`
      - `long_view = 1.0`
      - `like = 1.25`
      - `follow/comment/forward = 1.5`
    - time-decay half-life: `48h`
    - weighted mean pooling after a small content projection
  - long-term preference branch:
    - use recent `100` strong interactions from `strong_history_*`
    - build normalized `category_l1 + category_l2` distribution
    - `38 + 156 = 194` dims before projection
- Training:
  - label mode: `click_or_long`
  - loss: query-level sampled softmax
  - negatives:
    - `4` hard negatives
    - `24` random negatives
    - `0` explicit negatives
  - best checkpoint chosen by `valid hr@200`

### Training Result

- Best valid epoch: `5`
- Best valid metrics:
  - `hr@50 = 0.0895`
  - `hr@100 = 0.1389`
  - `hr@200 = 0.2064`
  - `ndcg@50 = 0.0381`
  - `ndcg@100 = 0.0507`
  - `ndcg@200 = 0.0656`

Observation:
- The learned content two-tower is much stronger than the previous zero-shot content route.
- It is still weaker than HSTU and weaker than the main dual-tower as a standalone route, but it now has meaningful complementary hits.

### Test Route Comparison

This comparison uses three learned/usable routes:
- `main` = dual-tower recall
- `content` = learned content two-tower
- `hstu` = HSTU sequential recall

#### Single-route test HR

| route | hr@50 | hr@100 | hr@200 |
| --- | ---: | ---: | ---: |
| main | 0.1149 | 0.1808 | 0.2720 |
| content two-tower | 0.0815 | 0.1221 | 0.1836 |
| hstu | 0.2066 | 0.3288 | 0.4867 |

Conclusion:
- Standalone route strength is `HSTU > main dual-tower > learned content two-tower`.

#### Query-level unique hit overlap at `@200`

- `main_only = 1071` (`5.61%`)
- `content_only = 574` (`3.01%`)
- `hstu_only = 4355` (`22.81%`)
- `main_hstu_only = 2260` (`11.84%`)
- `content_hstu_only = 1070` (`5.60%`)
- `all_three = 1607` (`8.42%`)

Interpretation:
- HSTU is not only the strongest route, it also contributes the largest block of unique hits.
- Main dual-tower still provides a non-trivial set of unique hits and should not be discarded.
- Learned content two-tower is weaker, but it is no longer redundant; it contributes stable unique-hit mass and is therefore worth keeping as a lightweight auxiliary route.

### Fixed-Budget Fusion Search

Search setup:
- total recall budget fixed to `200`
- candidate budgets searched on a `20`-step grid
- methods compared:
  - `weighted_rrf`
  - `append_hmc` (`hstu -> main -> content`)

Main findings:
- `weighted_rrf` is consistently better than plain append under the same budget split.
- If the objective is pure `hr@200`, `HSTU-only top200` is still better than any forced three-route split.
- Therefore, multi-route fusion is mainly useful for improving early precision (`@50`, `@100`) and for supplying diverse candidates to downstream rerank, not for maximizing raw `hr@200` under a hard fixed budget.

#### Best non-zero three-route split by metric

| target metric | method | hstu | main | content | result |
| --- | --- | ---: | ---: | ---: | ---: |
| best `hr@50` | `weighted_rrf` | 120 | 60 | 20 | 0.2128 |
| best `hr@100` | `weighted_rrf` | 140 | 40 | 20 | 0.3339 |
| best `hr@200` | `weighted_rrf` | 160 | 20 | 20 | 0.4518 |

Reference:
- `HSTU-only hr@200 = 0.4867`

### Recommended Default

For current multi-route recall before rerank, use:

- fusion method: `weighted_rrf`
- route budget: `HSTU : main : content = 160 : 20 : 20`
- equivalent weight ratio: `0.8 : 0.1 : 0.1`

Reason:
- It is the best non-zero three-route split on `hr@200`
- It still improves early precision compared with HSTU-only:
  - `hr@50`: `0.2097` vs `0.2066`
  - `hr@100`: `0.3330` vs `0.3288`
- It gives the smallest `hr@200` sacrifice among the searched three-route settings

Operational note:
- If the goal is pure recall-only `top200` without downstream rerank, keep `HSTU-only`.
- If the goal is to generate a diverse candidate set for rerank, use `weighted_rrf` with `160/20/20` as the current default.

## Weighted RRF Recall-Only Config 20260312

### Current Recall-Only Setup

- evaluation scope:
  - recall-only
  - no rerank metrics recorded in this entry
- route source depth: `200`
- selected fusion:
  - method: `weighted_rrf`
  - `rrf_k = 60`
  - budget split:
    - `hstu = 160`
    - `main = 20`
    - `content = 20`
- checkpoints:
  - main dual-tower:
    - `/home/hfx/KuaiRand/baseline/checkpoints/recall_pure/recall_model.pt`
  - learned content two-tower:
    - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_pure_v2/content_recall_model.pt`
  - hstu:
    - `/home/hfx/KuaiRand/baseline/checkpoints/hstu_recall_pure/hstu_recall_model.pt`

### Result Files

- route overlap / unique hit analysis:
  - `/home/hfx/KuaiRand/baseline/checkpoints/recall_route_overlap/test_overlap.json`
- fixed-budget fusion search:
  - `/home/hfx/KuaiRand/baseline/checkpoints/recall_route_overlap/fusion_search.json`

### Single-Route Recall Result

| route | hr@50 | hr@100 | hr@200 | ndcg@50 | ndcg@100 | ndcg@200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main dual-tower | 0.1149 | 0.1808 | 0.2720 | 0.0555 | 0.0724 | 0.0925 |
| learned content two-tower | 0.0815 | 0.1221 | 0.1836 | 0.0343 | 0.0447 | 0.0583 |
| hstu | 0.2066 | 0.3288 | 0.4867 | 0.0948 | 0.1261 | 0.1610 |

Conclusion:
- standalone route strength remains:
  - `HSTU > main dual-tower > learned content two-tower`

### Query-Level Complementarity at `@200`

- `main_only = 1071` (`5.61%`)
- `content_only = 574` (`3.01%`)
- `hstu_only = 4355` (`22.81%`)
- `main_hstu_only = 2260` (`11.84%`)
- `content_hstu_only = 1070` (`5.60%`)
- `all_three = 1607` (`8.42%`)

Interpretation:
- HSTU contributes the largest unique-hit block.
- main dual-tower still provides a meaningful unique complement.
- learned content two-tower is weaker, but not redundant.

### Fixed-Budget Fusion Search Result

Best non-zero three-route split by metric:

| target metric | method | hstu | main | content | result |
| --- | --- | ---: | ---: | ---: | ---: |
| best `hr@50` | `weighted_rrf` | 120 | 60 | 20 | 0.2128 |
| best `hr@100` | `weighted_rrf` | 140 | 40 | 20 | 0.3339 |
| best `hr@200` | `weighted_rrf` | 160 | 20 | 20 | 0.4518 |

Reference:
- `HSTU-only hr@200 = 0.4867`

### Selected Default for Current Recall-Only Experiments

- use `weighted_rrf`
- use route budget `160 / 20 / 20` (`hstu / main / content`)
- this is the current default when the goal is:
  - preserve route diversity
  - keep strong early precision
  - maintain a reasonable `hr@200` under fixed total budget `200`

Recall-only fused result for the selected config:

- `recall_hr@50 = 0.2097`
- `recall_hr@100 = 0.3330`
- `recall_hr@200 = 0.4518`
- `recall_ndcg@50 = 0.0975`
- `recall_ndcg@100 = 0.1291`
- `recall_ndcg@200 = 0.1558`

Operational note:
- if the goal is pure maximum `hr@200`, keep `HSTU-only top200`
- if the goal is multi-route recall-only under fixed budget, use `160/20/20 weighted_rrf`

## Content Two-Tower Signal-Positive Update 20260312

### Scope

- route:
  - learned content two-tower recall only
- label mode:
  - `signal_positive`
- evaluation protocol:
  - content-only recall
  - `main_topn = 0`
  - `content_topn = 200`
  - `hstu_topn = 0`
- comparison files:
  - old:
    - `/home/hfx/KuaiRand/baseline/checkpoints/compare_content/old.json`
  - new:
    - `/home/hfx/KuaiRand/baseline/checkpoints/compare_content/new.json`

### Checkpoints

- old checkpoint:
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v1/content_recall_model.pt`
- new checkpoint:
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v2_neg4_hard8_rand32_ep20/content_recall_model.pt`

### Change

New content two-tower training config:

- epochs:
  - `20`
- negative sampling:
  - `num_explicit_negatives = 4`
  - `num_hard_negatives = 8`
  - `num_random_negatives = 32`

Interpretation:

- this update mainly strengthens the query-level candidate set during training
- compared with the previous signal-positive content checkpoint, the new model learns against more realistic and more difficult negatives

### Result Comparison

#### Valid

| metric | old | new | delta |
| --- | ---: | ---: | ---: |
| `hr@50` | 0.0738 | 0.1013 | +0.0275 |
| `hr@100` | 0.1207 | 0.1586 | +0.0378 |
| `hr@200` | 0.1889 | 0.2369 | +0.0480 |
| `ndcg@50` | 0.0350 | 0.0445 | +0.0094 |
| `ndcg@100` | 0.0470 | 0.0591 | +0.0121 |
| `ndcg@200` | 0.0621 | 0.0765 | +0.0144 |

#### Test

| metric | old | new | delta |
| --- | ---: | ---: | ---: |
| `hr@50` | 0.0698 | 0.0901 | +0.0203 |
| `hr@100` | 0.1135 | 0.1449 | +0.0314 |
| `hr@200` | 0.1812 | 0.2166 | +0.0354 |
| `ndcg@50` | 0.0322 | 0.0406 | +0.0084 |
| `ndcg@100` | 0.0434 | 0.0547 | +0.0113 |
| `ndcg@200` | 0.0583 | 0.0705 | +0.0121 |

### Conclusion

- the updated negative sampling setup yields a clear and consistent gain on both valid and test
- the largest absolute improvement appears at `hr@200`
  - valid: `+0.0480`
  - test: `+0.0354`
- the learned content route is still weaker than HSTU and the main dual-tower as a standalone route, but it has become materially stronger than the previous signal-positive content checkpoint
- this version should replace the previous content signal-positive checkpoint as the default auxiliary learned content route in current recall experiments

## Content Two-Tower Wide v3 Update 20260312

### Scope

- route:
  - learned content two-tower recall only
- label mode:
  - `signal_positive`
- evaluation protocol:
  - content-only recall
  - `main_topn = 0`
  - `content_topn = 200`
  - `hstu_topn = 0`

### Checkpoints

- old checkpoint (`v2`):
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v2_neg4_hard8_rand32_ep20/content_recall_model.pt`
- new checkpoint (`v3`):
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v3_wide_neg4_hard8_rand32_ep20/content_recall_model.pt`

### Change

Compared with `v2`, `v3` keeps the same:

- epochs:
  - `20`
- negative sampling:
  - `num_explicit_negatives = 4`
  - `num_hard_negatives = 8`
  - `num_random_negatives = 32`

and only widens the tower:

- `embedding_dim: 16 -> 32`
- `tower_dim: 64 -> 128`
- `hidden_dim: 128 -> 256`

### Result Comparison

#### Valid

| metric | old | new | delta |
| --- | ---: | ---: | ---: |
| `hr@50` | 0.1013 | 0.1092 | +0.0079 |
| `hr@100` | 0.1586 | 0.1719 | +0.0133 |
| `hr@200` | 0.2369 | 0.2590 | +0.0221 |
| `ndcg@50` | 0.0445 | 0.0524 | +0.0079 |
| `ndcg@100` | 0.0591 | 0.0685 | +0.0093 |
| `ndcg@200` | 0.0765 | 0.0877 | +0.0113 |

#### Test

| metric | old | new | delta |
| --- | ---: | ---: | ---: |
| `hr@50` | 0.0901 | 0.1070 | +0.0169 |
| `hr@100` | 0.1449 | 0.1643 | +0.0194 |
| `hr@200` | 0.2166 | 0.2470 | +0.0304 |
| `ndcg@50` | 0.0406 | 0.0519 | +0.0113 |
| `ndcg@100` | 0.0547 | 0.0666 | +0.0119 |
| `ndcg@200` | 0.0705 | 0.0848 | +0.0144 |

### Conclusion

- widening the learned content two-tower on top of the stronger negative sampling setup yields another clear gain
- this gain is consistent across both valid and test, and again is largest at `hr@200`
  - valid: `+0.0221`
  - test: `+0.0304`
- the effect is smaller than the earlier negative-sampling upgrade, but still material
- therefore the current default learned content route should move from `v2` to `v3`

## Content Negative Immunity NegMix Gated Update 20260312

### Scope

- route:
  - learned content two-tower recall only
- label mode:
  - `signal_positive`
- evaluation protocol:
  - content-only recall
  - `main_topn = 0`
  - `content_topn = 200`
  - `hstu_topn = 0`

### Checkpoints

- old checkpoint (`v3` baseline):
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v3_wide_neg4_hard8_rand32_ep20/content_recall_model.pt`
- new checkpoint (`v5 negmix gated`):
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v5_negmix_gated_ep20/content_recall_model.pt`

### Change

Compared with `v3`, this experiment adds broader negative-feedback immunity on the learned content tower:

- strong negative feedback:
  - `is_hate == 1`
- weak negative feedback:
  - same-query losers from timesteps that contain positives
- filtering:
  - weak negatives are down-weighted and filtered by semantic similarity to the current positive-interest content representation
- immunity mode:
  - `gated`

Key config:

- `negative_history_len = 10`
- `negative_half_life_hours = 72`
- `negative_relative_weight = 0.35`
- `negative_hate_weight = 1.0`
- `negative_semantic_sim_threshold = 0.1`

### Result Comparison

#### Valid

| metric | old | new | delta |
| --- | ---: | ---: | ---: |
| `hr@50` | 0.1092 | 0.0993 | -0.0099 |
| `hr@100` | 0.1719 | 0.1567 | -0.0152 |
| `hr@200` | 0.2590 | 0.2347 | -0.0243 |
| `ndcg@50` | 0.0524 | 0.0455 | -0.0069 |
| `ndcg@100` | 0.0685 | 0.0602 | -0.0082 |
| `ndcg@200` | 0.0877 | 0.0775 | -0.0103 |
| `route_content_hr_recent_hate@200` | 0.2204 | 0.1953 | -0.0251 |
| `route_content_hr_no_recent_hate@200` | 0.2597 | 0.2354 | -0.0243 |
| `route_content_hate_l1_leakage@200` | 0.1003 | 0.0941 | -0.0062 |
| `route_content_hate_l2_leakage@200` | 0.0603 | 0.0639 | +0.0036 |

#### Test

| metric | old | new | delta |
| --- | ---: | ---: | ---: |
| `hr@50` | 0.1070 | 0.0922 | -0.0148 |
| `hr@100` | 0.1643 | 0.1471 | -0.0172 |
| `hr@200` | 0.2470 | 0.2208 | -0.0262 |
| `ndcg@50` | 0.0519 | 0.0427 | -0.0092 |
| `ndcg@100` | 0.0666 | 0.0568 | -0.0098 |
| `ndcg@200` | 0.0848 | 0.0730 | -0.0118 |
| `route_content_hr_recent_hate@200` | 0.2212 | 0.1903 | -0.0310 |
| `route_content_hr_no_recent_hate@200` | 0.2473 | 0.2212 | -0.0261 |
| `route_content_hate_l1_leakage@200` | 0.1029 | 0.0919 | -0.0110 |
| `route_content_hate_l2_leakage@200` | 0.0701 | 0.0672 | -0.0029 |

### Conclusion

- this `negmix + gated` setup should not replace `v3`
- the model becomes more suppressive and does reduce part of the category-level leakage
- however the suppression is too strong:
  - overall `HR/NDCG` drops on both valid and test
  - recent-negative samples also get worse instead of better
  - non-negative users are clearly hurt as well
- this suggests the current weak-negative mix is still too noisy and/or too heavily weighted for the learned content route
- next iteration should return to `hate-only` and use a more conservative immunity setup

## Content Negative Immunity Follow-Up 20260313

### Scope

- route:
  - learned content two-tower recall only
- goal:
  - test whether a more conservative explicit-negative immunity setup can improve recent-negative users without hurting global recall

### Variants Tried

#### 1. Conservative train-time immunity: `hate-only + fixed alpha`

- checkpoint:
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v6_hateonly_fixed_ep20/content_recall_model.pt`
- reference baseline:
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v3_wide_neg4_hard8_rand32_ep20/content_recall_model.pt`
- config:
  - `negative_immunity_mode = fixed`
  - `negative_fixed_alpha = 0.2`
  - `negative_relative_weight = 0.0`
  - `negative_hate_weight = 1.0`

Result summary:

- valid:
  - slight gain on `recent_hate` segment
  - slight drop on overall `hr@200`
    - `-0.0034`
- test:
  - overall metrics decline
  - `recent_hate` segment also declines
    - `route_content_hr_recent_hate@200 = -0.0133`

Conclusion:

- `hate-only + fixed alpha` is much safer than `negmix + gated`
- but it still does not produce a stable net gain
- therefore it should not replace `v3`

#### 2. Inference-time explicit-negative filtering

- no retraining
- base checkpoint:
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v3_wide_neg4_hard8_rand32_ep20/content_recall_model.pt`
- method:
  - first retrieve a larger candidate pool
  - use only explicit `is_hate == 1` history
  - hard-remove content candidates too similar to recent explicit negative content

Result summary:

- leakage decreases clearly:
  - both `hate_l1_leakage` and `hate_l2_leakage` go down
- but recent-negative recall also drops:
  - valid `route_content_hr_recent_hate@200 = -0.0161`
  - test `route_content_hr_recent_hate@200 = -0.0133`

Conclusion:

- explicit-negative filtering is working mechanically
- but the current hard-filter threshold is too aggressive
- it should not be kept as the default content-route behavior

### Overall Immunity Conclusion

- negative-feedback immunity has been explored in three forms:
  - `negmix + gated`
  - `hate-only + fixed`
  - inference-time hard filtering
- all three reduce part of the leakage signal
- none of them improves the final learned content route enough to justify replacing the current default
- current default remains:
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_twotower_recall_signal_positive_v3_wide_neg4_hard8_rand32_ep20/content_recall_model.pt`
  - with negative immunity disabled

## Main Dual-Tower Iterative Hard-Negative Mining Attempt 20260313

### Goal

- test an iterative mined-hard-negative curriculum for the main dual-tower

### Method

- warmup train a short main dual-tower checkpoint
- use that checkpoint to retrieve top items for each training query
- treat high-scoring non-positive items as mined hard negatives
- continue training with:
  - explicit negatives
  - rule-based hard negatives
  - mined hard negatives
  - random negatives

### Artifacts

- warmup checkpoint:
  - `/home/hfx/KuaiRand/baseline/checkpoints/recall_signal_positive_minehard_warmup_ep2/recall_model.pt`
- mined negatives:
  - `/home/hfx/KuaiRand/baseline/checkpoints/recall_signal_positive_minehard_warmup_ep2/train_mined_hard_negatives_ep2.npz`
- stage-2 checkpoint:
  - `/home/hfx/KuaiRand/baseline/checkpoints/recall_signal_positive_minehard_stage2_ep2/recall_model.pt`
- evaluation:
  - `/home/hfx/KuaiRand/baseline/checkpoints/recall_signal_positive_minehard_stage2_ep2/recall_metrics.json`

### Result

- stage-2 mined-hard result:
  - valid:
    - `hr@50 = 0.0931`
    - `hr@100 = 0.1504`
    - `hr@200 = 0.2350`
  - test:
    - `hr@50 = 0.0878`
    - `hr@100 = 0.1406`
    - `hr@200 = 0.2171`

### Conclusion

- this first mined-hard setup failed clearly
- the likely reasons are:
  - mining started too early from a weak warmup checkpoint
  - mined negatives were too many and too aggressive
  - the mined items likely contained many false negatives
- this approach is not kept in the current main dual-tower code path

## Final Three-Route Recall Version 20260313

### Scope

- task:
  - recall-only comparison and fusion
  - no end-to-end ranker training
- data version:
  - `KuaiRand-Pure`
- label mode:
  - `click_or_long`

### Final Route Checkpoints

- main dual-tower:
  - `/home/hfx/KuaiRand/baseline/checkpoints/recall_pure_refresh_ep20/recall_model.pt`
- learned content two-tower:
  - `/home/hfx/KuaiRand/baseline/checkpoints/content_recall_pure_refresh_ep20/content_recall_model.pt`
- HSTU route:
  - `/home/hfx/KuaiRand/baseline/checkpoints/hstu_recall_pure_bias_nostatic_ep20/hstu_recall_model.pt`

### HSTU Final Setting

- no user static profile features
- absolute position embedding:
  - enabled by the preprocessor
- relative position bias:
  - enabled
- relative time bias:
  - enabled

### Route Strength

From:

- `/home/hfx/KuaiRand/baseline/checkpoints/recall_three_routes_pure_refresh/diagnostic_200_200_200.json`

Single-route recall at `test`:

| route | `hr@50` | `hr@100` | `hr@200` |
| --- | ---: | ---: | ---: |
| main | 0.1446 | 0.2240 | 0.3306 |
| content | 0.0970 | 0.1509 | 0.2293 |
| hstu | 0.2085 | 0.3342 | 0.4922 |

Conclusion:

- route strength is still:
  - `HSTU > main > content`

### Complementarity

From:

- `/home/hfx/KuaiRand/baseline/checkpoints/recall_three_routes_pure_refresh/test_overlap.json`

At `test @200`:

- `main_only = 0.0589`
- `content_only = 0.0290`
- `hstu_only = 0.1870`
- `main_hstu_only = 0.1232`
- `content_hstu_only = 0.0519`
- `all_three = 0.1301`
- `any_hit = 0.5984`

Interpretation:

- HSTU provides the largest unique-hit block
- main dual-tower remains the most useful complementary route to HSTU
- content route is weaker, but still contributes non-trivial unique hits

### Fusion Search

From:

- `/home/hfx/KuaiRand/baseline/checkpoints/recall_three_routes_pure_refresh/fusion_search_valid.json`

After fixing the `valid/test` split bug in HSTU route lookup, the best weighted-RRF budgets on `valid` are:

- best `hr@50`:
  - `main / content / hstu = 60 / 20 / 120`
- best `hr@100`:
  - `main / content / hstu = 40 / 20 / 140`
- best `hr@200`:
  - `main / content / hstu = 20 / 20 / 160`

### Final Test Confirmation

Compared budgets:

- `20 / 20 / 160`
- `40 / 20 / 140`
- `60 / 20 / 120`

Test result:

| budget (`main/content/hstu`) | `hr@50` | `hr@100` | `hr@200` | `ndcg@50` | `ndcg@100` | `ndcg@200` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20 / 20 / 160` | 0.2121 | 0.3382 | 0.4582 | 0.0987 | 0.1310 | 0.1580 |
| `40 / 20 / 140` | 0.2164 | 0.3383 | 0.4409 | 0.1027 | 0.1339 | 0.1571 |
| `60 / 20 / 120` | 0.2162 | 0.3389 | 0.4217 | 0.1032 | 0.1347 | 0.1535 |

### Final Recommendation

If the goal is multi-route candidate generation for downstream rerank, use:

- fusion method:
  - `weighted_rrf`
- final budget:
  - `main / content / hstu = 20 / 20 / 160`

Reason:

- it gives the best `hr@200`
- it also gives the best `ndcg@200`
- it matches the route-strength structure:
  - strong HSTU backbone
  - small but useful main/content supplementation

Operational note:

- if the goal is pure recall-only maximum `hr@200`, `HSTU-only top200` is still stronger than the fused three-route result
- if the goal is a diverse multi-route candidate set before reranking, the current final version is:
  - `main = recall_pure_refresh_ep20`
  - `content = content_recall_pure_refresh_ep20`
  - `hstu = hstu_recall_pure_bias_nostatic_ep20`
  - fusion = `weighted_rrf`
  - budget = `20 / 20 / 160`
