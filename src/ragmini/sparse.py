"""BM25 稀疏召回。

不引第三方库，自己实现 Okapi BM25（30 行），原因是：
1. 面试问"BM25 的 k1/b 分别管什么"时要能答得出来，自己写过才答得实；
2. rank_bm25 对中文要先切词，包一层反而更绕。
"""

from __future__ import annotations

import math
from collections import Counter

import jieba

STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "及", "对", "为", "要", "应",
    "可以", "不能", "什么", "怎么", "多少", "哪些", "哪", "吗", "呢", "有",
    "一个", "进行", "按照", "根据", "通过", "以及", "这个", "那个",
}


def tokenize(text: str) -> list[str]:
    """jieba 精确模式 + 小写；去停用词，但保留数字与条款号。"""
    tokens: list[str] = []
    for raw in jieba.cut(text):
        token = raw.strip().lower()
        if not token or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in docs]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.n_docs = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.freqs = [Counter(toks) for toks in self.doc_tokens]

        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] += 1
        # 加一平滑，避免高频词 IDF 为负
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        scores = [0.0] * self.n_docs
        if not q_tokens or self.avgdl == 0:
            return scores
        for term in q_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:k] if score > 0]
