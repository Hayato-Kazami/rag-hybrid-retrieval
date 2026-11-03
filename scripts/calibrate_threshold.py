"""拒答阈值标定：不拍脑袋定一个数，用评测集扫出来。

做法：对每条问题记录"相关性首名分"，按可答/不可答分组看分布，
再在候选阈值上算误拒率与拒答准确率，取两者都可接受的点。

两种信号：
- dense  ：bge-m3 余弦，只衡量向量相近；
- rerank ：cross-encoder 的 sigmoid 分，直接建模 (query, passage) 相关性。

在"同领域、大量相似条款"的语料上，余弦的两组分布几乎重叠，
精排分才拉得开 —— 这个结论就是 docs/design.md 里"为什么用精排分判拒答"的依据。
"""

from __future__ import annotations

import argparse

from ragmini.config import Settings
from ragmini.evaluate import REJECT_WEIGHT, load_qa, relevance_scores
from ragmini.retrieve import Retriever


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", choices=["dense", "rerank"], default="rerank")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="逗号分隔；缺省时按信号给一组候选",
    )
    args = parser.parse_args()

    default_grid = {
        "dense": "0.30,0.40,0.50,0.55,0.58,0.60,0.62,0.65",
        "rerank": "0.05,0.10,0.20,0.30,0.50,0.70,0.85",
    }
    grid = args.thresholds or default_grid[args.signal]
    mode = "hybrid" if args.signal == "dense" else "hybrid+rerank"

    cfg = Settings()
    qa = load_qa(cfg.eval_file)
    retriever = Retriever(cfg=cfg).build()

    pos, neg = relevance_scores(retriever, qa, mode)
    print(f"信号={args.signal}  mode={mode}")
    print(f"可答   n={len(pos)}  min={min(pos):.3f} p10={percentile(pos,0.1):.3f} "
          f"median={percentile(pos,0.5):.3f} max={max(pos):.3f}")
    print(f"不可答 n={len(neg)}  min={min(neg):.3f} median={percentile(neg,0.5):.3f} "
          f"max={max(neg):.3f}")
    print()
    print(f"{'阈值':>6}{'误拒率':>9}{'拒答准':>9}{'综合':>9}")
    print("-" * 33)

    best = None
    for raw in grid.split(","):
        threshold = float(raw)
        false_reject = sum(1 for score in pos if score < threshold) / max(len(pos), 1)
        reject_acc = sum(1 for score in neg if score < threshold) / max(len(neg), 1)
        balanced = (1 - false_reject) * (1 - REJECT_WEIGHT) + reject_acc * REJECT_WEIGHT
        print(f"{threshold:>6.2f}{false_reject:>9.3f}{reject_acc:>9.3f}{balanced:>9.3f}")
        if best is None or balanced > best[1]:
            best = (threshold, balanced)

    if best:
        env_name = (
            "RAGMINI_ANSWER_THRESHOLD" if args.signal == "dense" else "RAGMINI_RERANK_THRESHOLD"
        )
        print(f"\n建议阈值：{best[0]:.2f}"
              f"（误拒权重 {1 - REJECT_WEIGHT:.1f} / 拒答权重 {REJECT_WEIGHT:.1f}）")
        print(f"写进 .env：{env_name}={best[0]:.2f}")
        print(
            "\n⚠️ 本脚本用的是全量评测集（in-sample）。"
            "要报样本外指标，用：python -m ragmini.cli eval --cv"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
