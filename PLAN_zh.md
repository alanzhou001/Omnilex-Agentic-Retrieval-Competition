# 项目实施计划

Kaggle **LLM Agentic Legal Information Retrieval**（大模型智能体法律信息检索）比赛的端到端实施计划。本计划基于 INTRO.md（比赛规则）、README.md（仓库配置）、CLAUDE.md（代码架构）与 ARCHITECTURE.md（模型方案）整理而来。

---

## 1. 目标与约束回顾

**目标**：给定一个英文法律问题，输出一组以分号分隔的瑞士法律引用（联邦法律 + BGE 联邦法院判决，通常以德语引用），在隐藏的标准答案集上最大化 **Macro F1** 得分。

**硬性约束**：
- Kaggle notebook **离线运行**（无互联网、无外部 API 调用）
- 在 Kaggle GPU（T4×2 或 P100）上运行时间 ≤ 12 小时
- 引用字符串必须与检索语料库中的字符串**完全匹配**
- 方案需可复现、可扩展（单样本推理成本 ≤ $10）

**可用数据**（已下载完成）：
- `data/llm-agentic-legal-information-retrieval/train.csv` — LEXam 训练集：查询 + 标准引用
- `data/llm-agentic-legal-information-retrieval/val.csv` — 验证集：查询 + 标准引用
- `data/llm-agentic-legal-information-retrieval/test.csv` — 测试集：仅查询
- `data/llm-agentic-legal-information-retrieval/laws_de.csv` — 德文联邦法律语料
- `data/llm-agentic-legal-information-retrieval/court_considerations.csv` — BGE 判决语料
- `data/llm-agentic-legal-information-retrieval/sample_submission.csv` — 提交格式示例

---

## 2. 方案总览

参照 [ARCHITECTURE.md](ARCHITECTURE.md)，整体流水线为：

```
查询（英文）
  → [可选] 英→德翻译
  → 双塔编码器（multilingual-e5-large，微调后）
  → 在语料库上执行 FAISS ANN 检索（top-100）
  → 交叉编码器重排序（mDeBERTa-v3-base，微调后）
  → 基于 Macro F1 校准的分数阈值
  → submission.csv
```

引用内容**从不由 LLM 直接生成** —— 始终从检索语料库中按原字符串复制（满足精确匹配要求）。

---

## 3. 分阶段计划

### 阶段 0 — 环境与数据准备（第 1 天）

**目标**：验证本地与 Kaggle 环境能够加载数据并完成一次平凡提交。

任务：
1. 执行 `pip install -e .` 与 `pip install -r requirements-dev.txt`
2. 运行 `pytest`，确认 citation normalizer 与 metrics 测试通过
3. 用 `pandas.head()`、`.info()` 查看每个 CSV 的形状与列名
4. 验证 `train.csv`/`val.csv` 中的标准引用字符串是否全部出现在语料库（`laws_de.csv` ∪ `court_considerations.csv`）中。计算覆盖率：训练/验证集标准引用中有多少比例在语料库中原样存在？若低于 100%，需排查不匹配原因 —— 该值决定了任何基于检索的方法的上限。
5. 产出一个**平凡基线提交**：对每条测试查询都预测训练集中最频繁的 top-5 标准引用。在本地用 `python scripts/evaluate_submission.py` 对验证集评分，确立下限。

交付物：
- `notebooks/00_data_exploration.ipynb`
- `submission_trivial.csv`（健全性检查用）
- 覆盖率报告（标准引用在语料库中的占比）

---

### 阶段 1 — 强基线：BM25 + 归一化（第 2-3 天）

**目标**：用一个无 LLM 的稳定纯检索系统，击败现有 notebook 基线。

任务：
1. 运行 `python scripts/build_indices.py`，在 `laws_de.csv` + `court_considerations.csv` 上构建 BM25 索引
2. 对每条验证查询：
   - 执行 BM25 检索（每个语料取 top-50）
   - 去重后按分数取 top-K
   - 原样输出引用（不经过 LLM 归一化）
3. 在验证集上调参 **K**（每条查询返回的引用数量），最大化 Macro F1。预期最优值在 K=3-10 之间 —— 引用过多会损害精度。
4. **跨语言探针**：同时使用离线的英→德翻译模型（如 `Helsinki-NLP/opus-mt-en-de`）对查询编码，与原查询的候选集合并。度量增量。
5. 保存为 `submission_bm25.csv`。

预期 Macro F1：~0.20-0.30

交付物：
- `notebooks/03_bm25_tuned.ipynb`
- `submission_bm25.csv`
- 调参表：验证集上 `K` 对应的 Macro F1

---

### 阶段 2 — 零样本稠密检索（第 4-5 天）

**目标**：在训练前先建立一个零样本多语言稠密检索基线。

任务：
1. 使用 `intfloat/multilingual-e5-large` 对整个语料（~271K 文档）做嵌入：
   - 文本模板：`"passage: {citation} | {title} | {text[:512]}"`（e5 要求语料前缀为 `passage:`，查询前缀为 `query:`）
   - Batch size 64，GPU；在 T4 上约 30-60 分钟
   - 保存为 `data/processed/corpus_embeddings.npy` + `corpus_ids.npy`
2. 构建 FAISS 索引：
   - 先试 `IndexFlatIP`（精确索引，271K × 1024 fp32 ≈ 1.1 GB 可承受）
   - 若太慢，改为 `IndexIVFFlat`，nlist=1024
3. 对每条验证查询取 top-100，在不同 K 截断下评估 Macro F1
4. **混合检索**：用倒数排名融合（RRF）合并 BM25 与稠密分数，通常优于单一方法。

预期 Macro F1：~0.35-0.45（零样本）；混合 RRF 可到 ~0.40-0.50。

交付物：
- `scripts/embed_corpus.py`
- `notebooks/04_dense_retrieval.ipynb`
- `data/processed/corpus_faiss.index`
- `submission_dense.csv`、`submission_hybrid.csv`

---

### 阶段 3 — 微调双塔编码器（第 6-8 天）

**目标**：在 LEXam 的 query/citation 配对上对双塔编码器做领域自适应。

任务：
1. **构造训练对**：
   - 正样本：(query, 标准引用对应的文档正文)，来自 train.csv
   - 难负样本：每条查询的 BM25 top-50 非标准结果（每个正样本配 3-5 个最难的）
2. **训练**：
   - 框架：`sentence-transformers`（对 e5 的封装较好）
   - 损失：`MultipleNegativesRankingLoss`（批内 + 显式难负样本的 InfoNCE）
   - 超参：bs=16-32、lr=2e-5、10% warmup、余弦衰减、3 个 epoch
   - 本地有 GPU 则本地训练；否则用 Colab Pro 或 Kaggle 训练 notebook
3. 使用微调后的编码器重新嵌入整个语料，重跑检索，在验证集上评估
4. 迭代：若提升停滞，可尝试更大 batch（梯度累积）、更长上下文，或换用 `BGE-M3` 作为备选骨干

预期 Macro F1：~0.50-0.60

交付物：
- `scripts/train_biencoder.py`
- `models/biencoder/`（微调后的 checkpoint）
- 新的语料嵌入 + FAISS 索引
- `submission_biencoder_ft.csv`

---

### 阶段 4 — 交叉编码器重排序（第 9-10 天）

**目标**：通过对双塔 top-100 做重排序，提升精度。

任务：
1. **构造重排训练对**：
   - 对每条训练查询，取微调双塔编码器的 top-100
   - 标签：在标准答案中记为 1，否则 0
   - 将负样本下采样至 ~5:1 的负正比
2. **训练**：
   - 模型：`microsoft/mdeberta-v3-base`（若许可允许也可用 `BAAI/bge-reranker-v2-m3`）
   - 输入：`"{query} [SEP] {citation} | {title} | {text[:400]}"`
   - 损失：BCE
   - bs=16、lr=1e-5、2 个 epoch
3. 推理流水线：
   - 双塔 → top-100 → 交叉编码器打分 → 降序排序
   - 阈值 θ：在验证集上网格搜索以最大化 Macro F1
4. **每查询自适应 K**：作为变体，预测需要返回的*引用数量*（例如基于交叉编码器分数分布的简单回归：top-1 分、与 top-2 的差距等特征）。在验证集上比较 fixed-θ 与 adaptive-K 谁更优。

预期 Macro F1：~0.60-0.70

交付物：
- `scripts/train_reranker.py`
- `models/reranker/` checkpoint
- `submission_reranked.csv`

---

### 阶段 5 — Kaggle 打包与提交（第 11-12 天）

**目标**：将训练好的流水线封装到一个离线可运行的 Kaggle notebook 中。

任务：
1. 作为 **Kaggle 数据集**上传：
   - 微调后的双塔编码器权重（~2.2 GB）
   - 微调后的重排序器权重（~350 MB）
   - 预计算的语料 FAISS 索引 + corpus_ids.npy（~1.1 GB）
   - `omnilex` 库源码（打包 zip）以便离线 `pip install`
2. 创建 `notebooks/90_submission.ipynb`：
   - Cell 1：`pip install` omnilex 与任何 wheel（离线路径来自 Kaggle 数据集）
   - Cell 2：从 `/kaggle/input/...` 加载双塔、重排序器、FAISS 索引
   - Cell 3：加载 `test.csv`
   - Cell 4：对每条查询 → 检索 → 重排 → 阈值 → 收集引用
   - Cell 5：以要求格式写出 `submission.csv`
3. 在 Kaggle 上以"Internet off"模式完整跑一遍
4. 先私密提交、迭代；稳定后再公开提交

交付物：
- Kaggle 数据集已上传
- `notebooks/90_submission.ipynb` 已提交
- 公开排行榜上有记录

---

### 阶段 6 — 消融与冲刺目标（若时间允许）

按预期收益排序：

1. **查询翻译**（英→德）后再做稠密检索，将英、德两套检索结果合并。
2. **引用图扩展**：若某 BGE 判决引用了 Art. X ZGB，则把该法条也加入候选集。在建索引时从语料文本构建图。
3. **LLM 重排 top-20**：使用小型离线 GGUF LLM（如 Qwen2.5-3B）作最后的 yes/no 相关性过滤。更慢但可能带来 2-5 个 F1 点的提升。
4. **集成**：用 val 上的特征做小型逻辑回归，学习 BM25 + 双塔 + 交叉编码器的分数融合权重。
5. 若时间允许，尝试 **ColBERT 风格的延迟交互** —— 召回率可能大幅提升。

---

## 4. 风险登记表

| 风险 | 缓解措施 |
|------|---------|
| 标准引用未全部原样出现在语料库中 | 阶段 0 的覆盖率分析；若差距较大，先排查归一化 bug 再改模型 |
| e5-large 与重排序器同时载入导致 Kaggle GPU OOM | 顺序加载（先用双塔、释放 GPU、再加载重排序器）。启用 fp16 |
| 运行时间超过 12h | 离线预计算语料嵌入；notebook 中只做查询编码与重排序 |
| 隐藏测试集分布与 LEXam 不同 | 最终架构确定后，用全量 train+val 合并训练；仅把验证集指标当作健全性检查，不要过拟合 θ |
| 交叉编码器成为运行时瓶颈 | 必要时将候选数从 top-100 降到 top-50 |
| 基座模型许可 | 推荐模型（e5、mDeBERTa、opus-mt）皆为 MIT/Apache/CC-BY 许可 —— 无虞 |

---

## 5. 代码结构新增

在现有仓库基础上新增以下文件：

```
scripts/
├── build_indices.py              # 已存在（BM25）
├── embed_corpus.py               # 新增：稠密语料嵌入
├── train_biencoder.py            # 新增：微调 e5
├── train_reranker.py             # 新增：微调 mDeBERTa
├── mine_hard_negatives.py        # 新增：BM25 top-k 非标准负样本
└── package_for_kaggle.py         # 新增：打包权重与上传辅助脚本

src/omnilex/
├── retrieval/
│   ├── bm25_index.py             # 已存在
│   ├── dense_index.py            # 新增：FAISS 封装 + e5 编码器
│   ├── hybrid.py                 # 新增：RRF 融合
│   └── reranker.py               # 新增：交叉编码器推理
├── training/
│   ├── __init__.py
│   ├── pair_builder.py           # 新增：(query, 正样本, 负样本) 数据集
│   └── losses.py                 # 新增：必要的损失封装
└── inference/
    ├── __init__.py
    └── pipeline.py               # 新增：端到端 Query → citations

notebooks/
├── 00_data_exploration.ipynb     # 新增
├── 03_bm25_tuned.ipynb           # 新增
├── 04_dense_retrieval.ipynb      # 新增
├── 05_biencoder_ft.ipynb         # 新增
├── 06_reranker_ft.ipynb          # 新增
└── 90_submission.ipynb           # 新增：最终可在 Kaggle 上提交的 notebook
```

---

## 6. 里程碑与成功标准

| 里程碑 | 目标 Macro F1（验证集） | 周次 |
|-------|-------------------------|------|
| M0 — 平凡基线 | 任意 > 0.0 | 1 |
| M1 — 调优后的 BM25 | ≥ 0.25 | 1 |
| M2 — 零样本稠密 | ≥ 0.40 | 2 |
| M3 — 双塔微调 | ≥ 0.55 | 2 |
| M4 — 完整流水线（双塔 + 交叉） | ≥ 0.65 | 2 |
| M5 — Kaggle 离线提交 | 同上，离线版 | 2 |
| M6 — 消融与冲刺 | ≥ 0.70 | 3+ |

每个里程碑是下一个的前置条件：**零样本没跑通前不做微调；双塔还不稳定时不上重排序**。一条能跑的简单流水线，胜过一条坏掉的复杂流水线。

---

## 7. 协作约定

- 每次检索运行都要产出 (a) `submission_<name>.csv` 与 (b) 写入 `runs.md` 一行记录：日期、脚本/notebook、验证 Macro F1、备注。留下审计轨迹。
- 所有训练脚本必须能用固定随机种子端到端复跑。
- 模型 checkpoint 与嵌入向量加入 git-ignore；仅提交代码 + 配置 + runs.md。
- Kaggle 提交 notebook 保持最小化 —— 所有重活（训练、嵌入）都在离线环境完成。
