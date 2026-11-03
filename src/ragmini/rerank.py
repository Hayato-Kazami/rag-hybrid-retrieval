"""重排。

cross-encoder 是主口径：把 (query, passage) 拼起来送进模型做二分类，
比双塔的余弦更准，代价是不能预计算 —— 所以只重排前 N 个候选。

lexical 是降级方案，不是等价替代：它只算词覆盖与数字命中，
在"问句里有明确数字/条款号"时接近 cross-encoder，在同义改写上差很多。
README 里的对比表把两者分开列了，不要混着说。
"""

from __future__ import annotations

import re
from typing import Protocol

from .sparse import tokenize

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """用 transformers 直接实现 cross-encoder，而不是依赖 CrossEncoder 封装。

    原因：封装层对目录结构有额外要求（要有 AutoProcessor 能识别的文件），
    而 bge-reranker 这类模型只提供 sentencepiece 词表；直接 load
    AutoModelForSequenceClassification 更通用，也更容易讲清白盒原理：
    把 (query, passage) 拼成一个序列，取 [CLS] 位置的 logit 过 sigmoid。
    """

    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.name = f"cross-encoder({model_name.replace(chr(92), '/').split('/')[-1]})"
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        torch = self._torch
        pairs = [[query, passage] for passage in passages]
        outputs: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                features = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                features = {key: value.to(self.device) for key, value in features.items()}
                logits = self.model(**features).logits.view(-1)
                outputs.extend(torch.sigmoid(logits.float()).cpu().tolist())
        return outputs


class LexicalReranker:
    """词覆盖 + 数字命中 + 长度惩罚的线性组合，仅用于无模型环境。"""

    name = "lexical"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q_tokens = set(tokenize(query))
        q_nums = set(NUM_RE.findall(query))
        scores: list[float] = []
        for passage in passages:
            p_tokens = set(tokenize(passage))
            coverage = len(q_tokens & p_tokens) / len(q_tokens) if q_tokens else 0.0
            p_nums = set(NUM_RE.findall(passage))
            num_hit = len(q_nums & p_nums) / len(q_nums) if q_nums else 0.0
            length_penalty = min(len(passage), 200) / 200 * 0.1
            scores.append(0.7 * coverage + 0.2 * num_hit + 0.1 * (1 - length_penalty))
        return scores


class NullReranker:
    name = "off"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.0 for _ in passages]


def build_reranker(name: str, model_name: str) -> Reranker:
    if name in {"off", "none"}:
        return NullReranker()
    if name == "lexical":
        return LexicalReranker()
    if name in {"cross-encoder", "cross_encoder", "ce"}:
        return CrossEncoderReranker(model_name)
    raise ValueError(f"未知 reranker: {name}")
