"""集中配置，全部可用环境变量覆盖，便于在不同机器上跑。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass
class Settings:
    corpus_dir: Path = ROOT / "data" / "corpus"
    eval_file: Path = ROOT / "data" / "eval" / "qa_68.jsonl"
    results_dir: Path = ROOT / "results"

    # dense 后端: bge-m3 | tfidf
    embedder: str = field(default_factory=lambda: _env_str("RAGMINI_EMBEDDER", "bge-m3"))
    embed_model: str = field(
        default_factory=lambda: _env_str("RAGMINI_EMBED_MODEL", "BAAI/bge-m3")
    )
    embed_batch: int = field(default_factory=lambda: _env_int("RAGMINI_EMBED_BATCH", 16))

    # rerank 后端: cross-encoder | lexical | off
    reranker: str = field(default_factory=lambda: _env_str("RAGMINI_RERANKER", "cross-encoder"))
    rerank_model: str = field(
        default_factory=lambda: _env_str("RAGMINI_RERANK_MODEL", "BAAI/bge-reranker-large")
    )
    rerank_top_n: int = field(default_factory=lambda: _env_int("RAGMINI_RERANK_TOP_N", 10))

    # 召回
    dense_top_k: int = field(default_factory=lambda: _env_int("RAGMINI_DENSE_TOP_K", 20))
    sparse_top_k: int = field(default_factory=lambda: _env_int("RAGMINI_SPARSE_TOP_K", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("RAGMINI_RRF_K", 60))

    # 拒答阈值
    # dense 余弦：无 rerank 时的兜底判据（同领域语料上区分度有限，见 docs/design.md）
    answer_threshold: float = field(
        default_factory=lambda: _env_float("RAGMINI_ANSWER_THRESHOLD", 0.60)
    )
    # 精排 sigmoid 分：有 rerank 时用它判"能不能答"，由 scripts/calibrate_threshold.py 标定
    # 68 题标定结果：阈值 0.70 时误拒 1/50、拒答 17/18
    rerank_threshold: float = field(
        default_factory=lambda: _env_float("RAGMINI_RERANK_THRESHOLD", 0.70)
    )

    # leaf 切分：语料里的条款平均 70 字，设 60 让长条真正切成两块，
    # 父子分块才不是摆设（命中 leaf、生成时回填整条）
    leaf_max_chars: int = field(default_factory=lambda: _env_int("RAGMINI_LEAF_MAX_CHARS", 60))


settings = Settings()
