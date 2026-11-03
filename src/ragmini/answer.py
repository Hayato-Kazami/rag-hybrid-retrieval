"""回答组装：上下文拼装 + 引用标注 + 可选 LLM 生成。

没有配置模型时走 extractive（直接给命中条款原文），并显式标注
`mode = "extractive"` —— 不给"看起来像模型生成"的假象。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .retrieve import Hit


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]
    mode: str          # llm | extractive | refused
    rejected: bool
    hits: list[Hit]


def format_citation(hit: Hit, index: int) -> str:
    title = hit.doc_title.split("（")[0]
    return f"[{index}] 《{title}》{hit.article}"


def build_context(hits: list[Hit], max_chars: int = 1200) -> str:
    blocks: list[str] = []
    budget = max_chars
    for idx, hit in enumerate(hits, 1):
        block = f"[{idx}] 《{hit.doc_title}》{hit.article}\n{hit.text}"
        if len(block) > budget:
            break
        blocks.append(block)
        budget -= len(block)
    return "\n\n".join(blocks)


SYSTEM_PROMPT = (
    "你是企业规章问答助手。只能依据给定资料回答，不得使用资料之外的知识。"
    "每条结论后用 [序号] 标注来源。资料不足以回答时，直接说“资料中没有相关规定”，"
    "不要推测，也不要编造条款号。"
)


def _llm_answer(question: str, context: str) -> str | None:
    api_key = os.getenv("RAGMINI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("RAGMINI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("RAGMINI_MODEL", "qwen-plus")
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content


def answer(question: str, hits: list[Hit], rejected: bool) -> Answer:
    citations = [format_citation(hit, i) for i, hit in enumerate(hits, 1)]

    if rejected or not hits:
        return Answer(
            question=question,
            text="资料中没有相关规定，无法回答这个问题。",
            citations=[],
            mode="refused",
            rejected=True,
            hits=hits,
        )

    context = build_context(hits[:5])
    generated = _llm_answer(question, context)
    if generated:
        return Answer(
            question=question,
            text=generated.strip(),
            citations=citations[:5],
            mode="llm",
            rejected=False,
            hits=hits,
        )

    top = hits[0]
    return Answer(
        question=question,
        text=f"（未配置生成模型，以下为命中条款原文）\n{top.text}",
        citations=citations[:5],
        mode="extractive",
        rejected=False,
        hits=hits,
    )
