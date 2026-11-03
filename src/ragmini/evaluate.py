"""检索侧评测：HitRate / MRR / nDCG + 拒答准确率。

只评检索，不评生成 —— 生成侧要 judge 模型，而 judge 的口径（用哪个模型、
温度多少、一致率怎么校准）是另一套工程，混在一起会让结论不可复现。
生成侧见 docs/design.md 的「还没做的」一节，不在此处假装有数字。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .retrieve import Retriever


@dataclass
class QA:
    qid: str
    question: str
    gold: list[str]
    qtype: str
    answerable: bool


def load_qa(path: Path) -> list[QA]:
    items: list[QA] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                QA(
                    qid=row["qid"],
                    question=row["question"],
                    gold=row.get("gold") or [],
                    qtype=row.get("type", "unknown"),
                    answerable=bool(row.get("answerable", True)),
                )
            )
    return items


def _first_hit_rank(hits, gold: list[str]) -> int | None:
    for rank, hit in enumerate(hits, 1):
        if hit.parent_id in gold:
            return rank
    return None


def _ndcg_at_k(hits, gold: list[str], k: int) -> float:
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], 1):
        if hit.parent_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
            break
    return dcg  # 只有一个相关文档，IDCG = 1


def evaluate(retriever: Retriever, qa: list[QA], mode: str, top_k: int = 10) -> dict:
    hit1 = hit5 = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    answerable_total = reject_total = reject_ok = false_reject = 0
    per_type: dict[str, dict[str, int]] = {}
    failures: list[dict] = []

    for item in qa:
        result = retriever.search(item.question, mode=mode, top_k=top_k)
        hits = result.hits

        if not item.answerable:
            reject_total += 1
            if result.rejected:
                reject_ok += 1
            continue

        answerable_total += 1
        if result.rejected:
            false_reject += 1

        rank = _first_hit_rank(hits, item.gold)
        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= 5:
                hit5 += 1
            mrr_sum += 1.0 / rank
        else:
            failures.append(
                {
                    "qid": item.qid,
                    "question": item.question,
                    "gold": item.gold,
                    "top1": hits[0].parent_id if hits else None,
                    "dense_top1": round(result.dense_top1, 4),
                }
            )
        ndcg_sum += _ndcg_at_k(hits, item.gold, 5)

        bucket = per_type.setdefault(item.qtype, {"total": 0, "hit5": 0})
        bucket["total"] += 1
        if rank is not None and rank <= 5:
            bucket["hit5"] += 1

    n = max(answerable_total, 1)
    return {
        "mode": mode,
        "top_k": top_k,
        "n_answerable": answerable_total,
        "n_unanswerable": reject_total,
        "hit_rate@1": round(hit1 / n, 4),
        "hit_rate@5": round(hit5 / n, 4),
        f"mrr@{top_k}": round(mrr_sum / n, 4),
        "ndcg@5": round(ndcg_sum / n, 4),
        "reject_accuracy": round(reject_ok / max(reject_total, 1), 4),
        "false_reject_rate": round(false_reject / n, 4),
        "by_type": {
            qtype: {
                "total": stats["total"],
                "hit_rate@5": round(stats["hit5"] / stats["total"], 4),
            }
            for qtype, stats in sorted(per_type.items())
        },
        "failures": failures,
    }


def evaluate_modes(
    retriever: Retriever,
    qa: list[QA],
    modes: list[str],
    top_k: int = 10,
) -> dict:
    reports = [evaluate(retriever, qa, mode=mode, top_k=top_k) for mode in modes]
    return {
        "embedder": retriever.embedder.name,
        "reranker": retriever.reranker.name,
        "n_questions": len(qa),
        "reports": reports,
    }
