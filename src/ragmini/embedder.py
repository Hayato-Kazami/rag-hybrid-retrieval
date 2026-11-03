"""向量化后端。

两个后端刻意做成同一接口，因为要回答"没有 GPU 能不能跑"这个问题：
- bge-m3:   真实语义向量，1024 维，效果是主口径；
- tfidf:    字符 2-3gram 的 TF-IDF，零下载、零显存，只用来验证链路通不通。
两者都做 L2 归一化，所以点积即余弦。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class BgeM3Embedder:
    """BAAI/bge-m3，支持中文长文本，1024 维。

    query 侧不加 instruction：bge-m3 的多语言检索不依赖指令前缀，
    加错前缀反而掉点（这是它和 bge-large-zh 的一个区别）。
    """

    name = "bge-m3"

    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class TfidfEmbedder:
    """字符 n-gram TF-IDF，纯 CPU。

    只作为离线冒烟与降级方案：它的"语义"来自字面重叠，
    所以长问句、同义改写会明显掉点 —— 这也正是需要混合检索与重排的原因。
    """

    name = "tfidf"

    def __init__(self, corpus_texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._normalize = normalize
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
        self.vectorizer.fit(corpus_texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return self._normalize(matrix).toarray().astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)


def build_embedder(name: str, corpus_texts: list[str], model_name: str, batch_size: int) -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder(corpus_texts)
    if name in {"bge-m3", "bge"}:
        return BgeM3Embedder(model_name, batch_size=batch_size)
    raise ValueError(f"未知 embedder: {name}")
