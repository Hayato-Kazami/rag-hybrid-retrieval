# rag-mini

一个**能跑通、能评测**的最小 RAG 闭环：条款级切分 → BGE-M3 向量 + BM25 双路召回 → RRF 融合 → cross-encoder 重排 → 引用溯源 → 检索评测。

不是又一个"文档问答 demo"：所有指标都在 `results/eval_report.json` 里，68 条评测集可复现，阈值是标定出来的而不是拍脑袋。

**在自建演示语料（3 篇规程 / 99 条 / 171 个检索块）上的实测结果：**

| 模式 | Hit@1 | Hit@5 | MRR | nDCG@5 | 拒答准确率 | 误拒率 |
|---|---|---|---|---|---|---|
| dense（BGE-M3 余弦） | 0.960 | 1.000 | 0.977 | 0.983 | 0.667 | 0.020 |
| sparse（BM25） | 0.900 | 0.960 | 0.930 | 0.938 | — | — |
| hybrid（RRF 融合） | 0.980 | 0.980 | 0.983 | 0.980 | 0.667 | 0.020 |
| **hybrid + rerank** | **0.980** | **1.000** | **0.987** | **0.990** | **0.944** | 0.020 |

三个能直接讲的结论：

1. **BM25 单独用会掉在改写型问题上**：`paraphrase` 类 Hit@5 只有 0.75，而 dense 是 1.00。
2. **RRF 融合会牺牲一点 Hit@5（1.000 → 0.980）换 Hit@1（0.960 → 0.980）**，因为名次融合会把个别原本第一的候选挤下去；重排再把它拉回来。
3. **拒答不能看余弦**：同领域语料上，可答与不可答的余弦分布在 0.56–0.62 重叠，拒答准确率只有 0.667；换成精排分数后两组分布是 0.23–0.86 与 0.95–1.00，拒答准确率 0.944。**排序分和"能不能答"是两件事。**

## 跑起来

```bash
pip install -e .
python -m ragmini.cli info                    # 看语料规模
python -m ragmini.cli ask "机器人电量低于多少要强制退出任务队列" --show-context
python -m ragmini.cli eval --modes dense,sparse,hybrid,hybrid+rerank
```

首次运行会下载 `BAAI/bge-m3`（约 2.2GB）与 `BAAI/bge-reranker-large`（约 1.3GB）。已有本地模型时直接指过去：

```bash
RAGMINI_EMBED_MODEL=/path/to/bge-m3 \
RAGMINI_RERANK_MODEL=/path/to/bge-reranker-large \
python -m ragmini.cli eval
```

**没有 GPU / 不想下模型**也能验证链路通不通（阈值要跟着换，字面向量的分数尺度完全不同）：

```bash
RAGMINI_EMBEDDER=tfidf RAGMINI_RERANKER=lexical \
RAGMINI_ANSWER_THRESHOLD=0.10 RAGMINI_RERANK_THRESHOLD=0.30 \
python -m ragmini.cli eval
```

实测检索仍能到 Hit@5 0.94–0.98，但**拒答基本失效**（准确率 0.06–0.78）——字面 TF-IDF 没有语义可比性，"仓库租金"和"仓库间距"的相似度都高。这个后端只用来验证链路，指标不能与 BGE-M3 混着报。

配置了 `RAGMINI_API_KEY`（DashScope 兼容模式）时生成走 LLM，否则返回命中条款原文，并在输出里标 `mode=extractive`——不给"看起来像模型生成"的假象。

## 为什么这么设计

| 选择 | 理由 |
|---|---|
| **按条款切，不定长切** | 条款是天然语义边界。跨条切会把两条不相干的义务混进一个块，检索命中后模型会串着答 |
| **父子分块** | 检索用 60 字以内的 leaf（小块命中率高），生成时回填整条（避免"一句话缺主语"） |
| **RRF 融合，不做分数加权** | 余弦和 BM25 分数不同量纲，加权求和要先归一化，又多一个超参；RRF 只用名次，少一个要调的东西 |
| **只重排 top-10** | cross-encoder 是逐对推理，全量重排延迟不可接受；实测前 10 已覆盖全部命中 |
| **拒答用精排分，不用余弦** | 余弦衡量"向量相近"，同领域条款天然相似；精排建模的是"这段材料能不能回答问题" |

更细的取舍与实测数据见 [`docs/design.md`](docs/design.md)。

## 评测方法

- **68 条**评测集：50 条可答（37 numeric / 8 paraphrase / 4 definition / 1 enum）+ 18 条不可答（人工构造的"库里没有"的问题）
- **Hit@k**：前 k 条里有没有出现 gold 条款（按条款 ID 匹配，不是按文本相似度）
- **MRR@10 / nDCG@5**：看首个命中的位置，不只看到没到
- **拒答准确率**：不可答问题中判为"资料不足"的比例
- **误拒率**：可答问题被误判为"资料不足"的比例
- 阈值用 `scripts/calibrate_threshold.py` 在评测集上扫出来，不手工指定

```bash
python scripts/calibrate_threshold.py --signal rerank
```

## 目录

```
src/ragmini/
  corpus.py     条款切分 + 父子分块
  embedder.py   BGE-M3 / TF-IDF 双后端（同接口）
  sparse.py     自实现 BM25（jieba 分词）
  index.py      numpy 向量索引（点积即余弦）
  rerank.py     cross-encoder（transformers 直连）/ 词覆盖降级
  retrieve.py   dense + sparse → RRF → rerank → 回填父块
  answer.py     上下文拼装 + 引用标注 + 可选 LLM
  evaluate.py   HitRate / MRR / nDCG / 拒答
data/corpus/    3 篇规程（虚构演示数据）
data/eval/      68 条评测集
scripts/        阈值标定
results/        评测报告（可复现）
```

## 数据说明

`data/corpus/` 里的三篇规程是**为这个实验虚构的演示数据**，不是任何真实企业制度或国家标准，数值均为编造。换成真实语料只需替换 md 文件，切分与评测代码不用改。

保留这份虚构语料是刻意的：真实项目里的手册与工单不能公开，而"链路能跑、指标可复现"这件事必须能被人验证。

## 还没做的

- **生成侧评测**（Faithfulness / Answer Relevancy）没做。要 judge 模型，而 judge 用哪个模型、温度多少、与人工标注的一致率怎么校准，是另一套工程；没有这些前提就报一个分数，等于把不可复现的东西当结论。
- **多路召回**（子问题改写、HyDE）只留了接口，没有实现——在当前语料上加了也看不出差异。
- **索引持久化**：每次启动重建，语料上万条后需要改成落盘 + 增量更新。
