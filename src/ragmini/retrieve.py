"""检索主链路：dense + sparse → RRF → rerank → 回溯父块。

一个刻意的设计：**融合用名次，不用分数**。
dense 的余弦和 BM25 的分数不在同一个量纲上，加权求和要先归一化，
而归一化本身又引入了新的超参；RRF 只用名次，1/(k+rank) 累加，
少一个要调的东西，也更稳。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, settings as default_settings
from .corpus import Corpus, load_corpus
from .embedder import Embedder, build_embedder
from .index import VectorIndex
from .rerank import Reranker, build_reranker
from .sparse import BM25


@dataclass
class Hit:
    chunk_id: str
    parent_id: str
    doc_id: str
    doc_title: str
    article: str
    leaf_text: str
    text: str
    score: float
    stage: str


@dataclass
class RetrievalResult:
    hits: list[Hit]
    dense_top1: float
    sparse_top1: float
    rejected: bool
    rerank_top1: float | None = None
    trace: dict = field(default_factory=dict)


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：同一文档在多路出现则累加。"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, doc_idx in enumerate(ranking, 1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class Retriever:
    def __init__(
        self,
        corpus: Corpus | None = None,
        cfg: Settings | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.cfg = cfg or default_settings
        self.corpus = corpus or load_corpus(self.cfg.corpus_dir, self.cfg.leaf_max_chars)
        self._embedder = embedder
        self._reranker = reranker
        self._index: VectorIndex | None = None
        self._bm25: BM25 | None = None

    # ---- 懒加载：评测要跑四组配置，避免重复加载模型 ----
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(
                self.cfg.embedder,
                self.corpus.leaf_texts(),
                self.cfg.embed_model,
                self.cfg.embed_batch,
            )
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = build_reranker(self.cfg.reranker, self.cfg.rerank_model)
        return self._reranker

    @property
    def index(self) -> VectorIndex:
        if self._index is None:
            vectors = self.embedder.encode(self.corpus.leaf_texts())
            self._index = VectorIndex(vectors)
        return self._index

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25(self.corpus.leaf_texts())
        return self._bm25

    def build(self) -> "Retriever":
        """显式预热，让首次查询不把加载时间算进延迟。"""
        _ = self.index
        _ = self.bm25
        return self

    # ---- 主入口 ----
    def search(self, question: str, mode: str = "hybrid+rerank", top_k: int = 5) -> RetrievalResult:
        leaves = self.corpus.leaves
        trace: dict = {}

        dense_rank: list[int] = []
        dense_scores: dict[int, float] = {}
        query_vector = self.embedder.encode_queries([question])[0]
        for idx, score in self.index.search(query_vector, self.cfg.dense_top_k):
            dense_rank.append(idx)
            dense_scores[idx] = score
        trace["dense_candidates"] = len(dense_rank)

        sparse_rank: list[int] = []
        sparse_scores: dict[int, float] = {}
        if mode != "dense":
            for idx, score in self.bm25.top_k(question, self.cfg.sparse_top_k):
                sparse_rank.append(idx)
                sparse_scores[idx] = score
            trace["sparse_candidates"] = len(sparse_rank)

        if mode == "dense":
            ordered = [(idx, dense_scores[idx]) for idx in dense_rank]
            stage = "dense"
        elif mode == "sparse":
            ordered = [(idx, sparse_scores[idx]) for idx in sparse_rank]
            stage = "sparse"
        else:
            ordered = rrf_fuse([dense_rank, sparse_rank], k=self.cfg.rrf_k)
            stage = "rrf"

        rerank_top1: float | None = None
        if mode.endswith("+rerank") and ordered:
            head = ordered[: self.cfg.rerank_top_n]
            passages = [leaves[idx].text for idx, _ in head]
            rerank_scores = self.reranker.score(question, passages)
            rerank_top1 = max(rerank_scores) if rerank_scores else None
            ordered = [
                (idx, score) for (idx, _), score in sorted(
                    zip(head, rerank_scores), key=lambda pair: pair[1], reverse=True
                )
            ]
            stage = f"{stage}+{self.reranker.name}"

        hits: list[Hit] = []
        for idx, score in ordered[:top_k]:
            leaf = leaves[idx]
            parent = self.corpus.get_parent(leaf.parent_id)
            hits.append(
                Hit(
                    chunk_id=leaf.chunk_id,
                    parent_id=leaf.parent_id,
                    doc_id=leaf.doc_id,
                    doc_title=leaf.doc_title,
                    article=leaf.article,
                    leaf_text=leaf.text,
                    text=parent.text if parent else leaf.text,
                    score=float(score),
                    stage=stage,
                )
            )

        dense_top1 = max(dense_scores.values()) if dense_scores else 0.0
        sparse_top1 = max(sparse_scores.values()) if sparse_scores else 0.0
        # 能不能回答，用"精排"分数判，不用余弦：
        # 余弦只衡量向量相近，同领域不同条款的相似度天然就高（实测 0.5–0.6），
        # 用它做阈值会在"问了个库里没有的问题"时大面积误答。
        # cross-encoder 直接建模 (query, passage) 的相关性，区分度明显更好。
        if mode == "sparse":
            rejected = False  # 稀疏模式没有可比的相关性分数，只作对照基线
        elif rerank_top1 is not None:
            rejected = rerank_top1 < self.cfg.rerank_threshold
        else:
            rejected = dense_top1 < self.cfg.answer_threshold

        trace.update({"stage": stage, "returned": len(hits)})
        return RetrievalResult(
            hits=hits,
            dense_top1=float(dense_top1),
            sparse_top1=float(sparse_top1),
            rejected=bool(rejected),
            rerank_top1=rerank_top1,
            trace=trace,
        )
