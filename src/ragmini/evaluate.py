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
    fold: str = "A"


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
                    fold=row.get("fold", "A"),
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


def _recall_at_k(hits, gold: list[str], k: int) -> float:
    """前 k 条覆盖了多少比例的正确文档。

    与 Hit@k 的差别只在 gold 多于 1 条时才显现：
    - gold 恒为 1 条时，本函数返回值必然是 0 或 1，与 Hit@k 完全相同；
    - gold 有 n 条时，命中 3 条就是 3/n，而 Hit@k 只要命中 1 条就算满分。
    所以这个指标的价值取决于评测集里有多少"多答案题"，见 evaluate() 返回的
    gold_size_dist —— 如果全是 1，加它只是多一个假指标。
    """
    if not gold:
        return 0.0
    got = {h.parent_id for h in hits[:k] if h.parent_id in gold}
    return len(got) / len(gold)


def evaluate(
    retriever: Retriever,
    qa: list[QA],
    mode: str,
    top_k: int = 10,
    thresholds_by_fold: dict[str, dict] | None = None,
) -> dict:
    hit1 = hit5 = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    recall5_sum = 0.0
    gold_size_dist: dict[int, int] = {}
    answerable_total = reject_total = reject_ok = false_reject = 0
    per_type: dict[str, dict[str, int]] = {}
    failures: list[dict] = []

    base_answer = retriever.cfg.answer_threshold
    base_rerank = retriever.cfg.rerank_threshold

    for item in qa:
        # 交叉验证时，这一条用的是"在另一折上选出来的阈值"——
        # 它自己的标签从未参与选择，所以这个判定是样本外的。
        override = (thresholds_by_fold or {}).get(item.fold)
        if override:
            retriever.cfg.answer_threshold = override["dense"]
            retriever.cfg.rerank_threshold = override["rerank"]
        else:
            retriever.cfg.answer_threshold = base_answer
            retriever.cfg.rerank_threshold = base_rerank

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

        gold_size_dist[len(item.gold)] = gold_size_dist.get(len(item.gold), 0) + 1
        recall5 = _recall_at_k(hits, item.gold, 5)
        recall5_sum += recall5

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

        bucket = per_type.setdefault(item.qtype, {"total": 0, "hit5": 0, "recall5": 0.0})
        bucket["total"] += 1
        if rank is not None and rank <= 5:
            bucket["hit5"] += 1
        bucket["recall5"] += recall5

    retriever.cfg.answer_threshold = base_answer
    retriever.cfg.rerank_threshold = base_rerank

    n = max(answerable_total, 1)
    # 所有 gold 都 ≤ 1 条时，Recall@5 与 Hit@5 恒等：两个数字不是互相印证的独立证据，
    # 报出来会让人以为做了两项校验。置这个标志，避免把同一件事数两遍。
    recall_equals_hit = all(size <= 1 for size in gold_size_dist)
    return {
        "mode": mode,
        "top_k": top_k,
        "n_answerable": answerable_total,
        "n_unanswerable": reject_total,
        "hit_rate@1": round(hit1 / n, 4),
        "hit_rate@5": round(hit5 / n, 4),
        "recall@5": round(recall5_sum / n, 4),
        "recall_equals_hit": recall_equals_hit,
        "gold_size_dist": dict(sorted(gold_size_dist.items())),
        f"mrr@{top_k}": round(mrr_sum / n, 4),
        "ndcg@5": round(ndcg_sum / n, 4),
        "reject_accuracy": round(reject_ok / max(reject_total, 1), 4),
        "false_reject_rate": round(false_reject / n, 4),
        "by_type": {
            qtype: {
                "total": stats["total"],
                "hit_rate@5": round(stats["hit5"] / stats["total"], 4),
                "recall@5": round(stats["recall5"] / stats["total"], 4),
            }
            for qtype, stats in sorted(per_type.items())
        },
        "failures": failures,
    }


DEFAULT_GRID = {
    "dense": [0.30, 0.40, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65],
    "rerank": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.85],
}
# 误拒比误答更能被用户感知：答得出来但被拒 = 功能不可用；答不出来硬答 = 幻觉。
# 所以权重偏向"别误拒"，0.6 / 0.4 是产品侧定的，不是数学最优。
REJECT_WEIGHT = 0.4


def relevance_scores(retriever: Retriever, qa: list[QA], mode: str, top_k: int = 10) -> tuple[list[float], list[float]]:
    """收集每条问题的"相关性首名分"，按可答/不可答分组。

    只取分数、不看 rejected，所以结果与当前阈值无关 —— 这是交叉标定能成立的前提。
    """
    pos: list[float] = []
    neg: list[float] = []
    use_rerank = mode.endswith("+rerank")
    for item in qa:
        result = retriever.search(item.question, mode=mode, top_k=top_k)
        if use_rerank:
            score = result.rerank_top1 if result.rerank_top1 is not None else 0.0
        else:
            score = result.dense_top1
        (pos if item.answerable else neg).append(score)
    return pos, neg


def pick_threshold(pos: list[float], neg: list[float], grid: list[float]) -> dict:
    """在候选阈值上扫一遍，按 0.6 / 0.4 的加权取最优点。"""
    best: dict | None = None
    for threshold in grid:
        false_reject = sum(1 for s in pos if s < threshold) / max(len(pos), 1)
        reject_acc = sum(1 for s in neg if s < threshold) / max(len(neg), 1)
        balanced = (1 - false_reject) * (1 - REJECT_WEIGHT) + reject_acc * REJECT_WEIGHT
        if best is None or balanced > best["balanced"]:
            best = {
                "threshold": threshold,
                "balanced": round(balanced, 4),
                "false_reject": round(false_reject, 4),
                "reject_accuracy": round(reject_acc, 4),
                "n_calibration": len(pos) + len(neg),
            }
    return best or {"threshold": 0.0, "balanced": 0.0, "false_reject": 0.0, "reject_accuracy": 0.0}


def cross_validate_thresholds(
    retriever: Retriever,
    qa: list[QA],
    mode: str,
    top_k: int = 10,
    grid: list[float] | None = None,
) -> dict[str, dict] | None:
    """两折交叉标定：每一折的判定阈值只用**另一折**选出来。

    为什么不是 3:1 切分：68 条切完 holdout 只剩 12 条可答 + 5 条不可答，
    1/5 的分辨率比结论本身还大。两折让每条都当过一次留出样本，
    最终指标仍基于全部 68 条，但每条用的阈值都来自另外半份数据。
    """
    if mode == "sparse":
        return None  # 稀疏分不可跨 query 比较，不做阈值标定
    folds = sorted({item.fold for item in qa})
    if len(folds) < 2:
        return None

    if grid is None:
        grid = DEFAULT_GRID["rerank" if mode.endswith("+rerank") else "dense"]

    out: dict[str, dict] = {}
    for fold in folds:
        calibration_set = [item for item in qa if item.fold != fold]
        pos, neg = relevance_scores(retriever, calibration_set, mode, top_k)
        picked = pick_threshold(pos, neg, grid)
        out[fold] = {
            "dense": picked["threshold"],
            "rerank": picked["threshold"],
            "calibrated_on": [f for f in folds if f != fold],
            **picked,
        }
    return out


def evaluate_modes(
    retriever: Retriever,
    qa: list[QA],
    modes: list[str],
    top_k: int = 10,
    protocol: str = "in-sample",
    fold_thresholds: dict[str, dict] | None = None,
) -> dict:
    reports = []
    for mode in modes:
        overrides = (fold_thresholds or {}).get(mode)
        report = evaluate(retriever, qa, mode=mode, top_k=top_k, thresholds_by_fold=overrides)
        if overrides:
            report["fold_thresholds"] = overrides
        reports.append(report)
    return {
        "embedder": retriever.embedder.name,
        "reranker": retriever.reranker.name,
        "n_questions": len(qa),
        "eval_protocol": protocol,
        "reports": reports,
    }
