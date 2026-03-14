#!/usr/bin/env python3
"""KuaiRand baseline dataset builder for two-stage recommendation.

Baseline design (fixed by project convention):
- Recall: plain two-tower retrieval.
- Ranker: single-task click ranking on exposure logs.
- Split: warmup/train/valid/test by date without random-exposure logs.
- Feature policy: only include online-available fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

import pandas as pd

"""
    原始日志读的属性及其字段类型:用户id 视频id 日期 小时分钟  时间戳 入口场景  点击标签和停留时长标签
"""
LOG_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "tab",
]

LOG_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "date": "int32",
    "hourmin": "int16",
    "time_ms": "int64",
    "is_click": "int8",
    "is_like": "int8",
    "is_follow": "int8",
    "is_comment": "int8",
    "is_forward": "int8",
    "is_hate": "int8",
    "long_view": "int8",
    "tab": "int8",
}

"""
    用户侧特征： uid、活跃等级、是否低活跃期、是否直播用户、是否视频作者、关注数、粉丝数、好友数、注册天数、一些加密离散特征
"""
USER_FEATURE_COLUMNS = [
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "follow_user_num_range",
    "fans_user_num",
    "fans_user_num_range",
    "friend_user_num",
    "friend_user_num_range",
    "register_days",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
]

"""
    视频侧特征: videoid 作者id 视频类型 上传方式 视频时长 配乐id 配乐类型 标签 视频宽和高
"""
ITEM_FEATURE_COLUMNS = [
    "video_id",
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "video_duration",
    "music_id",
    "music_type",
    "tag",
    "server_width",
    "server_height",
]
"""
    训练样本字段 除了上述原生字段的一些保留 还包括
    hourmin转小时桶
    日期映射序号
    当前样本之前的正反馈视频序列
    历史序列长度
"""
OUTPUT_INTERACTION_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "tab",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "hour_bucket",
    "date_bucket",
    "history_pos_video_ids",
    "history_pos_len",
    "history_click_video_ids",
    "history_click_len",
    "content_history_video_ids",
    "content_history_signal_types",
    "content_history_time_ms",
    "content_history_len",
    "strong_history_video_ids",
    "strong_history_signal_types",
    "strong_history_time_ms",
    "strong_history_len",
]


@dataclass(frozen=True)
class RawDatasetSpec:
    name: str
    warmup_log: str
    main_log: str
    user_features: str
    item_features: str
    excluded_logs: List[str]
    excluded_features: List[str]


RAW_DATASET_SPECS: Dict[str, RawDatasetSpec] = {
    "1k": RawDatasetSpec(
        name="1k",
        warmup_log="log_standard_4_08_to_4_21_1k.csv",
        main_log="log_standard_4_22_to_5_08_1k.csv",
        user_features="user_features_1k.csv",
        item_features="video_features_basic_1k.csv",
        excluded_logs=["log_random_4_22_to_5_08_1k.csv"],
        excluded_features=["video_features_statistic_1k.csv"],
    ),
    "pure": RawDatasetSpec(
        name="pure",
        warmup_log="log_standard_4_08_to_4_21_pure.csv",
        main_log="log_standard_4_22_to_5_08_pure.csv",
        user_features="user_features_pure.csv",
        item_features="video_features_basic_pure.csv",
        excluded_logs=["log_random_4_22_to_5_08_pure.csv"],
        excluded_features=["video_features_statistic_pure.csv"],
    ),
}


def resolve_raw_dataset_spec(data_dir: Path, dataset_version: str) -> RawDatasetSpec:
    if dataset_version != "auto":
        spec = RAW_DATASET_SPECS[dataset_version]
        missing = [
            name
            for name in (spec.warmup_log, spec.main_log, spec.user_features, spec.item_features)
            if not (data_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required {dataset_version} files under {data_dir}: {missing}"
            )
        return spec

    matched_specs = []
    for spec in RAW_DATASET_SPECS.values():
        required = (
            spec.warmup_log,
            spec.main_log,
            spec.user_features,
            spec.item_features,
        )
        if all((data_dir / name).exists() for name in required):
            matched_specs.append(spec)

    if len(matched_specs) == 1:
        return matched_specs[0]
    if len(matched_specs) > 1:
        raise ValueError(
            f"Multiple dataset variants matched under {data_dir}; "
            "please pass --dataset-version explicitly."
        )
    raise FileNotFoundError(
        f"Could not infer dataset variant from {data_dir}. "
        "Expected either *_1k.csv or *_pure.csv source files."
    )


@dataclass(frozen=True)
class DateSplit:
    """
    将数据切分
    warmup构建历史序列:20220408~20220421
    train:0422~0501
    valid:0502~0505
    test:0506~0508
    """
    name: str
    start_date: int
    end_date: int

    def contains(self, date: int) -> bool:
        return self.start_date <= date <= self.end_date


SPLITS = [
    DateSplit("warmup", 20220408, 20220421),
    DateSplit("train", 20220422, 20220501),
    DateSplit("valid", 20220502, 20220505),
    DateSplit("test", 20220506, 20220508),
]
#日期分桶
_DATE_BUCKET_MAP: Dict[int, int] = {}
_cur_dt = datetime.strptime("20220408", "%Y%m%d")
_end_dt = datetime.strptime("20220508", "%Y%m%d")
_idx = 0
while _cur_dt <= _end_dt:
    _DATE_BUCKET_MAP[int(_cur_dt.strftime("%Y%m%d"))] = _idx
    _cur_dt += timedelta(days=1)
    _idx += 1


@dataclass
class KuaiRandBaselineConfig:

    data_dir: Path
    output_dir: Path
    dataset_version: str = "auto"
    max_history_len: int = 500
    click_history_len: int = 20
    content_history_len: int = 15
    strong_history_len: int = 100
    sort_by_time: bool = True
    nrows_warmup: Optional[int] = None
    nrows_main: Optional[int] = None

    @property
    def raw_spec(self) -> RawDatasetSpec:
        return resolve_raw_dataset_spec(self.data_dir, self.dataset_version)

    @property
    def resolved_dataset_version(self) -> str:
        return self.raw_spec.name

    @property
    def warmup_log_path(self) -> Path:
        return self.data_dir / self.raw_spec.warmup_log

    @property
    def main_log_path(self) -> Path:
        return self.data_dir / self.raw_spec.main_log

    @property
    def user_feature_path(self) -> Path:
        return self.data_dir / self.raw_spec.user_features

    @property
    def item_feature_path(self) -> Path:
        return self.data_dir / self.raw_spec.item_features


class KuaiRandBaselineDatasetBuilder:
    """Build interaction samples and side feature tables for the baseline."""

    def __init__(self, config: KuaiRandBaselineConfig) -> None:
        self.config = config

    @staticmethod
    def split_from_date(date: int) -> Optional[str]:
        """
        给定日期 划分所在数据集
        """
        for split in SPLITS:
            if split.contains(int(date)):
                return split.name
        return None

    @staticmethod
    def date_bucket(date: int) -> int:
        """
        时间特征转桶
        """
        # Keep a continuous day index for downstream embedding.
        return _DATE_BUCKET_MAP.get(int(date), -1)

    @staticmethod
    def hour_bucket(hourmin: int) -> int:
        # Original field is HHMM encoded in int.
        return int(hourmin) // 100

    def _load_log(self, path: Path, nrows: Optional[int]) -> pd.DataFrame:
        """
        读取日志文件 按照时间顺序排序 并添加训练、验证、warmup标签
        """
        df = pd.read_csv(
            path,
            usecols=LOG_COLUMNS,
            dtype=LOG_DTYPES,
            nrows=nrows,
        )
        if self.config.sort_by_time:
            df = df.sort_values(["time_ms", "user_id", "video_id"], kind="mergesort")
        df["split"] = df["date"].map(self.split_from_date)
        df = df[df["split"].notnull()].reset_index(drop=True)
        return df

    def _iter_logs_in_time_order(self) -> Iterable[pd.Series]:
        for path, nrows in [
            (self.config.warmup_log_path, self.config.nrows_warmup),
            (self.config.main_log_path, self.config.nrows_main),
        ]:
            df = self._load_log(path, nrows)
            for row in df.itertuples(index=False):
                yield row

    def _write_side_tables(self) -> Dict[str, int]:
        """
        对于用户和视频 筛选需要的特征输出到新的文件
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        user_df = pd.read_csv(self.config.user_feature_path, usecols=USER_FEATURE_COLUMNS)
        item_df = pd.read_csv(self.config.item_feature_path, usecols=ITEM_FEATURE_COLUMNS)

        user_path = self.config.output_dir / "user_features.selected.csv"
        item_path = self.config.output_dir / "item_features.selected.csv"
        user_df.to_csv(user_path, index=False)
        item_df.to_csv(item_path, index=False)

        return {
            "user_rows": int(len(user_df)),
            "item_rows": int(len(item_df)),
        }

    def build(self) -> Dict[str, object]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        out_files = {
            "train": self.config.output_dir / "interactions.train.csv",
            "valid": self.config.output_dir / "interactions.valid.csv",
            "test": self.config.output_dir / "interactions.test.csv",
        }
        fhs = {k: v.open("w", newline="") for k, v in out_files.items()}
        writers = {k: csv.DictWriter(fhs[k], fieldnames=OUTPUT_INTERACTION_COLUMNS) for k in fhs}
        for writer in writers.values():
            writer.writeheader()

        user_histories: Dict[int, Deque[int]] = defaultdict(
            lambda: deque(maxlen=self.config.max_history_len)
        )
        click_histories: Dict[int, Deque[int]] = defaultdict(
            lambda: deque(maxlen=max(1, int(self.config.click_history_len)))
        )
        strong_histories: Dict[int, Deque[tuple[int, int, int]]] = defaultdict(
            lambda: deque(maxlen=max(self.config.max_history_len, self.config.strong_history_len))
        )
        weak_histories: Dict[int, Deque[tuple[int, int, int]]] = defaultdict(
            lambda: deque(maxlen=max(self.config.max_history_len, self.config.content_history_len))
        )

        stats = {
            "train": {"rows": 0, "click_sum": 0, "long_view_sum": 0},
            "valid": {"rows": 0, "click_sum": 0, "long_view_sum": 0},
            "test": {"rows": 0, "click_sum": 0, "long_view_sum": 0},
        }

        current_key: Optional[tuple[int, int]] = None
        current_rows: List[object] = []

        def event_signal_code(
            *,
            is_click: int,
            is_like: int,
            is_follow: int,
            is_comment: int,
            is_forward: int,
            is_hate: int,
            long_view: int,
        ) -> int:
            if is_follow == 1 or is_comment == 1 or is_forward == 1:
                return 4
            if is_like == 1:
                return 3
            if long_view == 1:
                return 2
            if is_click == 1 and is_hate == 0:
                return 1
            return 0

        def build_content_history(user_id: int) -> tuple[str, str, str, int]:
            target_len = max(0, int(self.config.content_history_len))
            if target_len <= 0:
                return "", "", "", 0

            selected: List[tuple[int, int, int]] = []
            seen_video_ids: set[int] = set()
            for history_events in (strong_histories[user_id], weak_histories[user_id]):
                for video_id, event_time_ms, signal_code in reversed(history_events):
                    if video_id in seen_video_ids:
                        continue
                    selected.append((int(video_id), int(signal_code), int(event_time_ms)))
                    seen_video_ids.add(int(video_id))
                    if len(selected) >= target_len:
                        break
                if len(selected) >= target_len:
                    break

            if not selected:
                return "", "", "", 0

            video_text = ",".join(str(video_id) for video_id, _, _ in selected)
            signal_text = ",".join(str(signal_type) for _, signal_type, _ in selected)
            time_text = ",".join(str(event_time_ms) for _, _, event_time_ms in selected)
            return video_text, signal_text, time_text, len(selected)

        def build_strong_history(user_id: int) -> tuple[str, str, str, int]:
            target_len = max(0, int(self.config.strong_history_len))
            if target_len <= 0:
                return "", "", "", 0

            selected = list(reversed(strong_histories[user_id]))
            if target_len > 0:
                selected = selected[:target_len]
            if not selected:
                return "", "", "", 0

            video_text = ",".join(str(int(video_id)) for video_id, _, _ in selected)
            signal_text = ",".join(str(int(signal_code)) for _, _, signal_code in selected)
            time_text = ",".join(str(int(event_time_ms)) for _, event_time_ms, _ in selected)
            return video_text, signal_text, time_text, len(selected)

        def flush_step(rows: List[object]) -> None:
            if not rows:
                return
            user_id = int(rows[0].user_id)
            history = user_histories[user_id]
            click_history = click_histories[user_id]
            history_text = ",".join(str(v) for v in history)
            history_len = len(history)
            click_history_text = ",".join(str(v) for v in click_history)
            click_history_len = len(click_history)
            positives_in_step: List[int] = []
            clicks_in_step: List[int] = []
            strong_in_step: List[tuple[int, int, int]] = []
            weak_in_step: List[tuple[int, int, int]] = []
            (
                content_history_video_ids,
                content_history_signal_types,
                content_history_time_ms,
                content_history_len,
            ) = build_content_history(user_id)
            (
                strong_history_video_ids,
                strong_history_signal_types,
                strong_history_time_ms,
                strong_history_len,
            ) = build_strong_history(user_id)

            for row in rows:
                split = row.split
                video_id = int(row.video_id)
                time_ms = int(row.time_ms)
                is_click = int(row.is_click)
                is_like = int(row.is_like)
                is_follow = int(row.is_follow)
                is_comment = int(row.is_comment)
                is_forward = int(row.is_forward)
                is_hate = int(row.is_hate)
                long_view = int(row.long_view)
                signal_code = event_signal_code(
                    is_click=is_click,
                    is_like=is_like,
                    is_follow=is_follow,
                    is_comment=is_comment,
                    is_forward=is_forward,
                    is_hate=is_hate,
                    long_view=long_view,
                )
                is_strong_positive = signal_code >= 2
                is_weak_positive = signal_code == 1
                if split in writers:
                    output = {
                        "user_id": user_id,
                        "video_id": video_id,
                        "date": int(row.date),
                        "hourmin": int(row.hourmin),
                        "time_ms": time_ms,
                        "tab": int(row.tab),
                        "is_click": is_click,
                        "is_like": is_like,
                        "is_follow": is_follow,
                        "is_comment": is_comment,
                        "is_forward": is_forward,
                        "is_hate": is_hate,
                        "long_view": long_view,
                        "hour_bucket": self.hour_bucket(int(row.hourmin)),
                        "date_bucket": self.date_bucket(int(row.date)),
                        "history_pos_video_ids": history_text,
                        "history_pos_len": history_len,
                        "history_click_video_ids": click_history_text,
                        "history_click_len": click_history_len,
                        "content_history_video_ids": content_history_video_ids,
                        "content_history_signal_types": content_history_signal_types,
                        "content_history_time_ms": content_history_time_ms,
                        "content_history_len": content_history_len,
                        "strong_history_video_ids": strong_history_video_ids,
                        "strong_history_signal_types": strong_history_signal_types,
                        "strong_history_time_ms": strong_history_time_ms,
                        "strong_history_len": strong_history_len,
                    }
                    writers[split].writerow(output)
                    stats[split]["rows"] += 1
                    stats[split]["click_sum"] += is_click
                    stats[split]["long_view_sum"] += long_view
                if signal_code > 0:
                    positives_in_step.append(video_id)
                if is_click == 1:
                    clicks_in_step.append(video_id)
                if is_strong_positive:
                    strong_in_step.append((video_id, time_ms, signal_code))
                elif is_weak_positive:
                    weak_in_step.append((video_id, time_ms, signal_code))

            # Same-timestep samples must share the same pre-step history.
            # Only after the whole step is written do we update the user history.
            for video_id in positives_in_step:
                history.append(video_id)
            for video_id in clicks_in_step:
                click_history.append(video_id)
            for video_id, event_time_ms, signal_code in strong_in_step:
                strong_histories[user_id].append((video_id, event_time_ms, signal_code))
            for video_id, event_time_ms, signal_code in weak_in_step:
                weak_histories[user_id].append((video_id, event_time_ms, signal_code))

        for row in self._iter_logs_in_time_order():
            row_key = (int(row.user_id), int(row.time_ms))
            if current_key is None:
                current_key = row_key
            if row_key != current_key:
                flush_step(current_rows)
                current_rows = []
                current_key = row_key
            current_rows.append(row)

        flush_step(current_rows)

        for fh in fhs.values():
            fh.close()

        for split, split_stats in stats.items():
            if split_stats["rows"] > 0:
                split_stats["click_rate"] = split_stats["click_sum"] / split_stats["rows"]
                split_stats["long_view_rate"] = (
                    split_stats["long_view_sum"] / split_stats["rows"]
                )
            else:
                split_stats["click_rate"] = 0.0
                split_stats["long_view_rate"] = 0.0

        side_table_stats = self._write_side_tables()

        metadata = {
            "dataset_version": self.config.resolved_dataset_version,
            "raw_data_dir": str(self.config.data_dir),
            "raw_input_files": {
                "warmup_log": self.config.warmup_log_path.name,
                "main_log": self.config.main_log_path.name,
                "user_features": self.config.user_feature_path.name,
                "item_features": self.config.item_feature_path.name,
            },
            "split_rule": {
                "warmup": "20220408-20220421",
                "train": "20220422-20220501",
                "valid": "20220502-20220505",
                "test": "20220506-20220508",
            },
            "history_positive_rule": (
                "signal_code > 0 "
                "(follow/comment/forward OR like OR long_view OR click without hate)"
            ),
            "click_history_rule": {
                "history_len": int(self.config.click_history_len),
                "source": "recent exposures with is_click == 1 only",
                "same_step_policy": "same user_id + time_ms exposures share the same pre-step click history",
                "output_field": "history_click_video_ids",
            },
            "content_history_rule": {
                "history_len": int(self.config.content_history_len),
                "strong_positive": [
                    "long_view == 1",
                    "is_like == 1",
                    "is_follow == 1",
                    "is_comment == 1",
                    "is_forward == 1",
                ],
                "weak_positive": [
                    "is_click == 1",
                    "not strong_positive",
                    "is_hate == 0",
                ],
                "signal_codes": {
                    "click": 1,
                    "long_view": 2,
                    "like": 3,
                    "follow_or_comment_or_forward": 4,
                },
                "selection_priority": "recent strong positives first, then recent weak positives",
            },
            "strong_history_rule": {
                "history_len": int(self.config.strong_history_len),
                "source": "recent strong positives only",
                "signal_codes": {
                    "long_view": 2,
                    "like": 3,
                    "follow_or_comment_or_forward": 4,
                },
            },
            "max_history_len": self.config.max_history_len,
            "click_history_len": int(self.config.click_history_len),
            "excluded_logs": self.config.raw_spec.excluded_logs,
            "excluded_features": self.config.raw_spec.excluded_features,
            "feature_schema": {
                "log_input_columns": LOG_COLUMNS,
                "user_feature_columns": USER_FEATURE_COLUMNS,
                "item_feature_columns": ITEM_FEATURE_COLUMNS,
                "interaction_output_columns": OUTPUT_INTERACTION_COLUMNS,
            },
            "interaction_stats": stats,
            "side_table_stats": side_table_stats,
        }

        with (self.config.output_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KuaiRand baseline data for two-tower + single-task click ranker."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/KuaiRand-Pure/data"),
        help="Path of KuaiRand raw data directory.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="pure",
        choices=["auto", "1k", "pure"],
        help="Raw dataset variant. Defaults to pure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hfx/KuaiRand/baseline/processed_pure"),
        help="Output directory for processed csv files.",
    )
    parser.add_argument(
        "--max-history-len",
        type=int,
        default=500,
        help="Maximum positive-history length kept per user.",
    )
    parser.add_argument(
        "--click-history-len",
        type=int,
        default=20,
        help="Maximum recent click-only history length exported for ranker training.",
    )
    parser.add_argument(
        "--content-history-len",
        type=int,
        default=15,
        help="Maximum recent high-quality history events exported for content recall.",
    )
    parser.add_argument(
        "--strong-history-len",
        type=int,
        default=100,
        help="Maximum recent strong-positive history events exported for lightweight long-term preference.",
    )
    parser.add_argument(
        "--disable-sort",
        action="store_true",
        help="Disable time sorting within each source file.",
    )
    parser.add_argument(
        "--nrows-warmup",
        type=int,
        default=None,
        help="Debug option: only read first N rows from warmup log.",
    )
    parser.add_argument(
        "--nrows-main",
        type=int,
        default=None,
        help="Debug option: only read first N rows from main log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = KuaiRandBaselineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
        max_history_len=args.max_history_len,
        click_history_len=args.click_history_len,
        content_history_len=args.content_history_len,
        strong_history_len=args.strong_history_len,
        sort_by_time=not args.disable_sort,
        nrows_warmup=args.nrows_warmup,
        nrows_main=args.nrows_main,
    )
    builder = KuaiRandBaselineDatasetBuilder(config)
    metadata = builder.build()
    print(json.dumps(metadata["interaction_stats"], indent=2))


if __name__ == "__main__":
    main()
