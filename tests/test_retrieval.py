"""离线测试：只用 tfidf + lexical，不下载任何模型，保证 CI 上秒级跑完。"""

from __future__ import annotations

import pytest

from ragmini.config import Settings
from ragmini.corpus import load_corpus
from ragmini.evaluate import load_qa
from ragmini.retrieve import Retriever, rrf_fuse


@pytest.fixture(scope="module")
def corpus():
    cfg = Settings()
    return load_corpus(cfg.corpus_dir, cfg.leaf_max_chars)


@pytest.fixture(scope="module")
def retriever(corpus):
    cfg = Settings(embedder="tfidf", reranker="lexical", answer_threshold=0.99)
    return Retriever(corpus=corpus, cfg=cfg).build()


def test_corpus_structure(corpus):
    assert len({leaf.doc_id for leaf in corpus.leaves}) >= 3
    assert len(corpus.parents) >= 50
    # 父子分块：leaf 数量应多于条数
    assert len(corpus.leaves) >= len(corpus.parents)


def test_rrf_prefers_doc_in_both_lists():
    fused = rrf_fuse([[1, 2, 3], [3, 4, 5]], k=60)
    assert fused[0][0] == 3


def test_answerable_query_hits_gold(retriever):
    result = retriever.search("两台机器人互相等待超过多久算死锁？", mode="hybrid", top_k=5)
    assert result.hits
    assert any(hit.parent_id == "demo-01-dispatch#第4条" for hit in result.hits)


def test_numeric_query_hits_gold(retriever):
    result = retriever.search("易燃液体库房温度不能超过多少度？", mode="hybrid", top_k=5)
    assert any(hit.parent_id == "demo-02-hazmat#第8条" for hit in result.hits)


def test_reject_when_no_relevant_chunk(retriever):
    # 阈值被调到 0.99，任何问题都会判为不足以回答，用于验证分支逻辑
    result = retriever.search("公司的年假有几天？", mode="hybrid", top_k=5)
    assert result.rejected is True


def test_eval_set_is_balanced():
    qa = load_qa(Settings().eval_file)
    assert len(qa) == 68
    assert sum(1 for item in qa if not item.answerable) == 18
