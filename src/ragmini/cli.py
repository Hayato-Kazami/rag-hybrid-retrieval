"""命令行入口。

python -m ragmini.cli ask "机器人电量低于多少要强制退出任务队列"
python -m ragmini.cli eval
python -m ragmini.cli info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer import answer as build_answer
from .config import settings
from .evaluate import cross_validate_thresholds, evaluate_modes, load_qa
from .retrieve import Retriever


def cmd_ask(args: argparse.Namespace) -> int:
    retriever = Retriever().build()
    result = retriever.search(args.question, mode=args.mode, top_k=args.top_k)
    reply = build_answer(args.question, result.hits, result.rejected)

    print(f"\n问题：{reply.question}")
    print(f"模式：{reply.mode}   检索：{result.trace.get('stage')}   拒答：{reply.rejected}")
    print(f"\n回答：\n{reply.text}")
    if reply.citations:
        print("\n引用：")
        for line in reply.citations:
            print(f"  {line}")
    if args.show_context:
        print("\n候选（含分数）：")
        for rank, hit in enumerate(result.hits, 1):
            print(f"  {rank}. {hit.parent_id}  score={hit.score:.4f}  {hit.leaf_text[:40]}...")
    print()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    retriever = Retriever().build()
    qa = load_qa(settings.eval_file)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    protocol = "in-sample"
    fold_thresholds: dict[str, dict] = {}
    if args.cv:
        protocol = "2-fold-cross-validated"
        print("\n交叉标定：每一折的阈值只用另一折选（每条问题都是样本外判定）")
        for mode in modes:
            folds = cross_validate_thresholds(retriever, qa, mode, top_k=args.top_k)
            if not folds:
                continue
            fold_thresholds[mode] = folds
            detail = "  ".join(
                f"{fold}:{info['threshold']:.2f}（标定于 {'/'.join(info['calibrated_on'])}，"
                f"误拒 {info['false_reject']:.2f} / 拒答 {info['reject_accuracy']:.2f}）"
                for fold, info in folds.items()
            )
            print(f"  {mode:<16}{detail}")

    report = evaluate_modes(
        retriever,
        qa,
        modes=modes,
        top_k=args.top_k,
        protocol=protocol,
        fold_thresholds=fold_thresholds or None,
    )

    print(f"\nembedder={report['embedder']}  reranker={report['reranker']}  "
          f"n={report['n_questions']}  protocol={report['eval_protocol']}")
    header = f"{'mode':<22}{'Hit@1':>8}{'Hit@5':>8}{'MRR':>8}{'nDCG@5':>9}{'拒答准':>8}{'误拒':>8}"
    print(header)
    print("-" * len(header))
    for row in report["reports"]:
        print(
            f"{row['mode']:<22}{row['hit_rate@1']:>8.3f}{row['hit_rate@5']:>8.3f}"
            f"{row[f'mrr@{args.top_k}']:>8.3f}{row['ndcg@5']:>9.3f}"
            f"{row['reject_accuracy']:>8.3f}{row['false_reject_rate']:>8.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["reports"] and report["reports"][0].get("recall_equals_hit"):
        print(
            "\n注：本评测集每题 gold 均 ≤ 1 条，Recall@5 与 Hit@5 恒等"
            "（报告里两个字段数值相同，不是两项独立校验）。"
            "要真正测覆盖率，需补「多 gold」的清单类题。"
        )
    print(f"\n报告已写入 {out_path}")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    retriever = Retriever()
    corpus = retriever.corpus
    docs = sorted({leaf.doc_id for leaf in corpus.leaves})
    print(f"\n语料文件：{len(docs)} 篇")
    for doc_id in docs:
        leaves = [leaf for leaf in corpus.leaves if leaf.doc_id == doc_id]
        title = leaves[0].doc_title
        parents = len({leaf.parent_id for leaf in leaves})
        print(f"  - {doc_id}  {title}  条数={parents}  leaf={len(leaves)}")
    print(f"\nleaf 总数：{len(corpus.leaves)}（平均 {sum(len(l.text) for l in corpus.leaves)/len(corpus.leaves):.0f} 字）")
    print(f"评测集：{settings.eval_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragmini", description="最小可评测 RAG 闭环")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="问一个问题")
    ask.add_argument("question", help="自然语言问题")
    ask.add_argument("--mode", default="hybrid+rerank",
                     choices=["dense", "sparse", "hybrid", "hybrid+rerank", "dense+rerank"])
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--show-context", action="store_true", help="打印候选与分数")
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("eval", help="跑评测集")
    ev.add_argument("--modes", default="dense,hybrid,hybrid+rerank")
    ev.add_argument("--top-k", type=int, default=10)
    ev.add_argument(
        "--cv",
        action="store_true",
        help="两折交叉标定：阈值只在另一折上选，指标是样本外结果",
    )
    ev.add_argument("--out", default="results/eval_report.json")
    ev.set_defaults(func=cmd_eval)

    info = sub.add_parser("info", help="看语料规模")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
