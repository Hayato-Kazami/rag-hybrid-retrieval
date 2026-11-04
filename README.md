# rag-hybrid-retrieval

一个**能跑通、能评测**的最小 RAG 闭环：条款级切分 → BGE-M3 向量 + BM25 双路召回 → RRF 融合 → cross-encoder 重排 → 引用溯源 → 检索评测。

不是又一个"文档问答 demo"：所有指标都在 `results/eval_report.json` 里，68 条评测集可复现，拒答阈值用**两折交叉验证**标定——每条问题的判定阈值都来自另外半份数据，不是拿它自己的标签调出来的。

**在自建演示语料（3 篇规程 / 99 条 / 171 个检索块）上的实测结果（拒答列为两折交叉验证的样本外结果）：**

| 模式 | Hit@1 | Hit@5 | MRR | nDCG@5 | 拒答准确率 | 误拒率 |
|---|---|---|---|---|---|---|
| dense（BGE-M3 余弦） | 0.960 | 1.000 | 0.977 | 0.983 | 0.889 | 0.040 |
| sparse（BM25） | 0.900 | 0.960 | 0.930 | 0.938 | — | — |
| hybrid（RRF 融合） | 0.980 | 0.980 | 0.983 | 0.980 | 0.889 | 0.040 |
| **hybrid + rerank** | **0.980** | **1.000** | **0.987** | **0.990** | **0.889** | **0.020** |

### 阈值策略对拒答指标的影响（同一份数据，三种口径）

| 阈值策略 | dense | hybrid+rerank | 说明 |
|---|---|---|---|
| 固定阈值（代码里的默认 0.60 / 0.70） | 0.667 | **0.944** | 0.60 并不是 dense 的最优点，0.70 是全量标定出来的 |
| 两折交叉（样本外，主报告） | 0.889 | **0.889** | 每条问题的阈值来自另一折，是能对外报的数 |
| 全量标定（in-sample 上界） | 0.944 | 0.944 | 阈值和指标同源，只能当上界看 |

这张表本身就是一个结论：**0.944 与 0.889 的差，就是阈值过拟合的幅度**（18 条不可答样本里差 1 条）。

三个能直接讲的结论：

1. **BM25 单独用会掉在改写型问题上**：`paraphrase` 类 Hit@5 只有 0.75，而 dense 是 1.00。
2. **RRF 融合会牺牲一点 Hit@5（1.000 → 0.980）换 Hit@1（0.960 → 0.980）**，因为名次融合会把个别原本第一的候选挤下去；重排再把它拉回来。
3. **拒答信号真正的差别不在准确率，在阈值稳定性**：交叉验证后两者同为 0.889（18 条里差 1 条，统计上不显著），但达到这个水平所需的精度完全不同——dense 的阈值从 0.60 挪到 0.65，拒答率就从 0.667 跳到 0.944，代价是误拒翻倍到 0.040；rerank 在 0.70 附近是一段平台。**dense 不是不能判拒答，是它对 0.02 量级的阈值误差没有容错。**

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
| **拒答优先用精排分** | 余弦也能判（交叉验证下两者同为 0.889），但它要求阈值精确到 0.02 量级——0.60 与 0.65 就差 5 条，工程上不敢这么调；精排分的阈值是一段平台，容错更高 |

更细的取舍与实测数据见 [`docs/design.md`](docs/design.md)。

## 评测方法

- **68 条**评测集：50 条可答（37 numeric / 8 paraphrase / 4 definition / 1 enum）+ 18 条不可答（人工构造的"库里没有"的问题）
- **Hit@k**：前 k 条里有没有出现 gold 条款（按条款 ID 匹配，不是按文本相似度）
- **Recall@k**：前 k 条覆盖了多大比例的正确条款。**但本评测集每题 gold 都是 1 条，它与 Hit@k 在数学上恒等**（代码会输出 `recall_equals_hit` 标志，为 true 时两个字段数值必然相同），所以不能当成第二项独立证据——原因见「评测局限」
- **MRR@10 / nDCG@5**：看首个命中的位置，不只看到没到
- **拒答准确率**：不可答问题中判为"资料不足"的比例
- **误拒率**：可答问题被误判为"资料不足"的比例
- **两折交叉标定**：评测集按 `type × answerable` 分层切成 A/B 两折（A 35 条 / B 33 条，各含 9 条不可答）。A 折的判定阈值只用 B 折选，B 折只用 A 折选，然后逐条判定。**每条问题的阈值都没见过它自己的标签。**

```bash
# 主报告：两折交叉（样本外）
python -m ragmini.cli eval --cv --modes dense,sparse,hybrid,hybrid+rerank

# 对照：固定阈值（in-sample），用于量化过拟合幅度
python -m ragmini.cli eval --modes dense,sparse,hybrid,hybrid+rerank \
    --out results/eval_report_insample.json

# 只看某一个信号的阈值曲线
python scripts/calibrate_threshold.py --signal rerank
```

## 评测局限

写在这里，是因为这些限制比指标本身更值得知道：

- **只有 68 条**。不可答 18 条，差 1 条就是 5.6 个百分点——表里 0.944 与 0.889 的差距，样本量并不足以说它稳定。
- **分类型看，`enum` 只有 1 条**，`by_type` 里那一行不构成结论。
- **语料是 99 条虚构条款**。同领域真实法规的条款密度、交叉引用、长句结构都不一样，指标不能直接外推到真实语料。
- **没有多 gold 题，因此测不出 Recall**。50 条可答题的 gold 长度**全是 1**（`gold_size_dist` 实测为 `{"1": 50}`），连唯一的 `enum` 题也只标了 1 条。当 gold 恒为 1 条时 `Recall@k = |命中|/1`，取值只能是 0 或 1，与 `Hit@k` 完全等价——**加这个指标不会带来任何新信息**。要真正回答"该找的条款找全了没"，得把清单类问题标成多 gold；这是数据集问题，不是代码问题。
- **只评检索，不评生成**。Hit@5 高不代表答案对——生成侧需要 judge 模型和人工一致性校准，那是另一套工程。
- **两折交叉仍有残余偏差**：折的划分由我按类型分层手工决定，不是随机重复 K 次取均值。严格做法是 5 折 × 多次重复，68 条撑不住。

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
data/eval/      68 条评测集（带 fold 字段，供两折交叉标定）
scripts/        阈值标定
results/
  eval_report.json          主报告：两折交叉验证（样本外）
  eval_report_insample.json 对照：固定阈值（用于量化过拟合幅度）
```

## 数据说明

`data/corpus/` 里的三篇规程是**为这个实验虚构的演示数据**，不是任何真实企业制度或国家标准，数值均为编造。换成真实语料只需替换 md 文件，切分与评测代码不用改。

保留这份虚构语料是刻意的：真实项目里的手册与工单不能公开，而"链路能跑、指标可复现"这件事必须能被人验证。

## 还没做的

- **生成侧评测**（Faithfulness / Answer Relevancy）没做。要 judge 模型，而 judge 用哪个模型、温度多少、与人工标注的一致率怎么校准，是另一套工程；没有这些前提就报一个分数，等于把不可复现的东西当结论。
- **多路召回**（子问题改写、HyDE）只留了接口，没有实现——在当前语料上加了也看不出差异。
- **索引持久化**：每次启动重建，语料上万条后需要改成落盘 + 增量更新。

## License

MIT —— 见 [LICENSE](LICENSE)。`data/corpus/` 下的演示语料为虚构数据，可自由替换或删除。
