"""索引：全量向量放内存，点积即余弦。

为什么不上 FAISS：语料是几百到几万条的规模，numpy 一次 matmul 就是毫秒级，
多一个原生依赖不值得。上量之后换 FAISS/Milvus 只需要替换这个类的 search 方法。
"""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维矩阵")
        self.vectors = vectors.astype(np.float32)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self.vectors @ query_vector.astype(np.float32)
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def search_batch(self, query_vectors: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        results: list[list[tuple[int, float]]] = []
        for row in query_vectors:
            results.append(self.search(row, k))
        return results
