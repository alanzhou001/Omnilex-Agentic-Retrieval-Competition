# Proposed Model Architecture: Dense Retrieval + Cross-Encoder Reranking

## Problem Framing

Given an English query, retrieve the correct subset of ~271K Swiss legal citations (federal laws + court decisions, mostly German). Scored on Macro F1 — citation set overlap, not ranking. The key challenge is cross-lingual retrieval (English → German) and exact citation matching.

The baseline uses BM25 + a local GGUF LLM in a ReAct loop. The main weaknesses:
- BM25 is lexical — poor cross-lingual recall
- LLM-generated citations hallucinate; only citations extracted from retrieved docs are safe
- 3-iteration agent loop is slow and brittle

---

## Proposed Architecture: Bi-Encoder + Cross-Encoder Pipeline

```
Query (EN)
    │
    ▼
[Query Encoder]  ←── multilingual-e5-large or LaBSE
    │                 fine-tuned on (query, citation_text) pairs
    │
    ▼
ANN Search over corpus embeddings  (FAISS IVF index)
    │  top-100 candidates
    ▼
[Cross-Encoder Reranker]  ←── mDeBERTa-v3-base
    │                          fine-tuned on (query, citation_text) → relevance score
    │  top-20 reranked
    ▼
[Threshold / Calibration]
    │  keep citations above score threshold θ (tuned on val set for F1)
    ▼
Predicted citation set  →  submission.csv
```

---

## Base Models

### Bi-Encoder (retrieval)
**`intfloat/multilingual-e5-large`** (560M params)

- Trained on 1B+ multilingual pairs; strong EN↔DE zero-shot performance
- Produces 1024-dim embeddings; fits in Kaggle GPU memory
- Alternative: `sentence-transformers/LaBSE` (470M) — slightly weaker but faster

### Cross-Encoder (reranker)
**`microsoft/mdeberta-v3-base`** (86M params)

- Multilingual DeBERTa; strong on NLI/relevance tasks
- Takes `[CLS] query [SEP] citation_text [SEP]` → scalar relevance score
- Small enough to rerank 100 candidates per query within time budget

---

## Key Architectural Changes for This Task

### 1. Citation text construction
Each corpus document has `text`, `title`, `citation` fields. Concatenate for encoding:
```
"{citation} | {title} | {text[:512]}"
```
This puts the citation string at the front so the cross-encoder sees it early.

### 2. Cross-lingual query expansion (optional, cheap)
Translate query to German with a small MT model (Helsinki-NLP/opus-mt-en-de, 74M params) before encoding. Run both EN and DE queries, merge candidate sets. Improves recall ~5-10% on cross-lingual benchmarks.

### 3. Threshold calibration for F1
Macro F1 rewards variable-length prediction sets. After reranking, tune threshold θ on the training set:
- For each query, keep all citations with score > θ
- Grid search θ ∈ [0.1, 0.9] to maximize macro F1 on held-out validation split
- A per-query adaptive threshold (based on score gap) can further improve F1

### 4. Hard negative mining for bi-encoder fine-tuning
BM25 negatives are too easy. Mine hard negatives by:
1. Run BM25 retrieval on training queries
2. Take top-50 BM25 results that are NOT gold citations → hard negatives
3. Fine-tune bi-encoder with in-batch negatives + hard negatives (InfoNCE loss)

---

## Training Plan

### Data
- **Positives**: LEXam training set — (query, gold_citation_text) pairs
- **Negatives**: BM25 top-50 non-gold results per query (hard negatives)
- **Split**: 80/10/10 train/val/test from LEXam

### Stage 1: Fine-tune Bi-Encoder (~2-4 hours on A100)
```
Loss: MultipleNegativesRankingLoss (InfoNCE)
Batch size: 32 (with in-batch negatives)
Hard negatives per positive: 3
LR: 2e-5, warmup 10%, cosine decay
Epochs: 3
Max seq length: 512 (query), 256 (citation)
```

### Stage 2: Fine-tune Cross-Encoder (~1-2 hours on A100)
```
Loss: Binary cross-entropy (relevant=1, non-relevant=0)
Positives: gold citations
Negatives: bi-encoder top-100 non-gold (harder than BM25)
Ratio: 1:5 pos:neg
LR: 1e-5, warmup 6%
Epochs: 2
Max seq length: 512 total
```

### Stage 3: Threshold calibration (minutes)
```
Grid search θ on validation set
Metric: macro F1
```

---

## Kaggle Submission Constraints

| Component | Size | Notes |
|-----------|------|-------|
| multilingual-e5-large | ~2.2 GB | Upload as Kaggle dataset |
| mdeberta-v3-base | ~350 MB | Upload as Kaggle dataset |
| FAISS index (corpus embeddings) | ~1.1 GB (271K × 1024 × float32) | Pre-build, upload as dataset |
| opus-mt-en-de (optional) | ~300 MB | Upload as Kaggle dataset |

Total: ~4 GB, well within Kaggle dataset limits. Inference time estimate:
- Bi-encoder query encoding: <1s per query
- FAISS ANN search: <10ms per query
- Cross-encoder reranking (100 candidates): ~2-5s per query on GPU
- For ~500 test queries: ~30-45 min total — well within 12h limit

---

## Expected Improvements Over Baseline

| System | Expected Macro F1 | Notes |
|--------|-------------------|-------|
| BM25 baseline | ~0.20-0.30 | Lexical only, no cross-lingual |
| Bi-encoder (zero-shot) | ~0.35-0.45 | multilingual-e5 out of the box |
| Bi-encoder (fine-tuned) | ~0.50-0.60 | Domain-adapted on LEXam |
| + Cross-encoder reranker | ~0.60-0.70 | Better precision |
| + Query translation | ~0.65-0.75 | Better recall on German terms |

Estimates are rough; actual numbers depend on test set difficulty.

---

## Alternative / Ablation Ideas

- **ColBERT-style late interaction**: better recall than bi-encoder, but larger index (~10x)
- **Citation graph expansion**: if a retrieved BGE decision cites a statute, add that statute as a candidate
- **LLM reranker**: use a small instruct LLM (Qwen2.5-3B) to score top-20 candidates — slower but potentially higher precision
- **Ensemble**: combine BM25 scores + bi-encoder scores + cross-encoder scores with learned weights
