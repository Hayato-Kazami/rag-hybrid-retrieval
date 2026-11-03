"""语料加载与切分。

切分策略（README 里也写了取舍）：
1. 先按法条结构切（`## 第N条`），不做定长切片 —— 条款是天然的语义边界，
   跨条切会把两条不相干的义务混进同一块。
2. 条内再按句切 leaf（<= leaf_max_chars），检索命中的是 leaf。
3. 命中后回溯到整条（parent）喂给模型，避免"一句话里缺主语"。
   这就是常说的父子分块：小块检索、大块生成。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ARTICLE_RE = re.compile(r"^##\s*(第\d+条)\s*(.*)$")
H1_RE = re.compile(r"^#\s+(.*)$")
SENT_SPLIT = re.compile(r"(?<=[。；！？])")


@dataclass
class Leaf:
    chunk_id: str
    parent_id: str
    doc_id: str
    doc_title: str
    article: str
    text: str


@dataclass
class Parent:
    parent_id: str
    doc_id: str
    doc_title: str
    article: str
    title: str
    text: str


@dataclass
class Corpus:
    leaves: list[Leaf] = field(default_factory=list)
    parents: dict[str, Parent] = field(default_factory=dict)

    def leaf_texts(self) -> list[str]:
        return [leaf.text for leaf in self.leaves]

    def get_parent(self, parent_id: str) -> Parent | None:
        return self.parents.get(parent_id)


def _split_sentences(text: str, max_chars: int) -> list[str]:
    """按句拼接到不超过 max_chars；单句超长时硬切。"""
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(sent), max_chars):
                chunks.append(sent[i : i + max_chars])
            continue
        if len(buf) + len(sent) <= max_chars:
            buf += sent
        else:
            if buf:
                chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)
    return chunks or [text]


def _parse_doc(path: Path, max_chars: int) -> tuple[list[Leaf], dict[str, Parent]]:
    doc_id = path.stem
    lines = path.read_text(encoding="utf-8").splitlines()

    doc_title = doc_id
    body_start = 0
    for i, line in enumerate(lines):
        m = H1_RE.match(line)
        if m:
            doc_title = m.group(1).strip()
            body_start = i + 1
            break

    # 条目聚合
    articles: list[tuple[str, str, list[str]]] = []
    current: tuple[str, str, list[str]] | None = None
    for line in lines[body_start:]:
        m = ARTICLE_RE.match(line)
        if m:
            if current:
                articles.append(current)
            current = (m.group(1), m.group(2).strip(), [])
            continue
        if current is not None:
            stripped = line.strip()
            if stripped and not stripped.startswith(">"):
                current[2].append(stripped)
    if current:
        articles.append(current)

    leaves: list[Leaf] = []
    parents: dict[str, Parent] = {}
    for article, title, body_lines in articles:
        text = "".join(body_lines).strip()
        if not text:
            continue
        parent_id = f"{doc_id}#{article}"
        parents[parent_id] = Parent(
            parent_id=parent_id,
            doc_id=doc_id,
            doc_title=doc_title,
            article=article,
            title=title,
            text=text,
        )
        for idx, piece in enumerate(_split_sentences(text, max_chars)):
            leaves.append(
                Leaf(
                    chunk_id=f"{parent_id}#{idx}",
                    parent_id=parent_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    article=article,
                    text=piece,
                )
            )
    return leaves, parents


def load_corpus(corpus_dir: Path, max_chars: int = 120) -> Corpus:
    corpus = Corpus()
    for path in sorted(corpus_dir.glob("*.md")):
        leaves, parents = _parse_doc(path, max_chars)
        corpus.leaves.extend(leaves)
        corpus.parents.update(parents)
    if not corpus.leaves:
        raise FileNotFoundError(f"语料目录为空: {corpus_dir}")
    return corpus
