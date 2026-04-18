# LLM Agentic Legal Information Retrieval

Open for everyone! Retrieve the right Swiss legal sources for each question. Optimize citation-level F1 on a hidden test set.

## Overview

This competition challenges you to build an LLM-powered [agentic](https://en.wikipedia.org/wiki/AI_agent) legal information retrieval system for Swiss law.

For each English legal question, your model must retrieve and output the most relevant Swiss legal sources (statutes, decisions, etc. mostly cited in German). The goal is to produce a set of citations that best matches the reference solution on a hidden test set.

You’ll get a public training dataset to develop your approach, and you’ll be evaluated on how accurately your predicted citations match the ground truth (citation-level Macro F1).

## Description

Legal research is ultimately an **information retrieval** problem: given a legal question, find the **most relevant sources** (statutes, cases, and other sources) that support the answer. In this competition, you will build an **LLM-powered (agentic) retrieval pipeline** for **Swiss law**.

### The task

For each query, output a **semicolon-separated list of citations** to the most relevant Swiss legal sources. Queries are written in **English**, while cited sources are often referenced in **German** (as in real Swiss practice). Your system can be any approach (BM25, vector embedding retrieval, reranking, multi-step agentic retrieval, citation-graph based, transfer learning, hybrid systems, etc.) as long as it produces the correct citation list in the required format.

This is a **code competition**: You will have to submit a Kaggle notebook that can reproduce your `submission.csv` in an offline environment (no internet). Notebooks intentionally create realistic constraints similar to production settings (bounded compute and privacy-sensitive domain). It has to run in at most [12 hours, which is a constraint given by Kaggle](https://www.kaggle.com/docs/notebooks#technical-specifications).
**Note**: Submitting just a `submission.csv` is fine for testing but does not qualify for prizes. If you want to be eligible for prizes you will have to share your notebook with the host or publicly AFTER THE END of the competition and I recommend already submitting notebooks privately during the competition. 

### Data

We provide three categories of data:

- **Training set:** [**LEXam**](https://huggingface.co/datasets/LEXam-Benchmark/LEXam/viewer/open_question) (Fan et al., 2025) open-question queries with gold citations extracted from "answer" field.
  - Source: https://huggingface.co/datasets/LEXam-Benchmark/LEXam/viewer/open_question
  - Paper: https://arxiv.org/abs/2505.12864
  - License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
    https://creativecommons.org/licenses/by/4.0/
  - No endorsement: The LEXam authors and Hugging Face do not endorse or sponsor this competition.
- **Test set:** queries only; final ranking is based on a **hidden** set of similar queries. The process used for creating this data is confidential, please do not publicly speculate about it.
- **Retrieval corpus:** Swiss federal laws and federal court decision considerations (the searchable sources)
  - Under Swiss law, official enactments and official decisions/reports of authorities are not protected by copyright ([Swiss Copyright Act, Art. 5](https://www.fedlex.admin.ch/eli/cc/1993/1798_1798_1798/en#art_5)). The excerpts are provided for research/competition purposes; no guarantee of completeness or official status.

See the "Data" tab for more details.

### Practical notes

- **Citations matter:** small formatting differences can cause mismatches. We use the citations from the Retrieval corpus. If you generate citations with an LLM they may have small mismatches.
- **No leakage:** the hidden test set is designed to avoid exact overlap with the training data and public data. So using all data and models accessible to you is allowed as long as you comply with the "Rules". See the "Rules" tab for further details.
- **Reproducibility:** top solutions may be reviewed; winners may be asked to share a short solution write-up if not provided in the submitted Notebook.

### Evaluation

Submissions are scored using [**Macro F1**](https://en.wikipedia.org/wiki/F-score#:~:text=[29]-,Macro F1) computed per-query between your predicted citations and our gold citation set, then averaged across queries. This rewards systems that are adaptive to be consistently accurate across the full test distribution. For example the gold citation set may include more citations for more challenging legal questions.

- **Public leaderboard:** 50% of test queries (feedback during the competition)
- **Private leaderboard:** 50% of test queries (final ranking)

------

## Submission File

For each query_id in the test set, you must predict the set of gold_citations separated by ;. The file should contain a header and have the following format:

```tap
query_id,predicted_citations
test_001,"Art. 111 ZGB;Art. 114 ZGB;BGE 119 II 449 E. 3.4"
test_002,"BGE 116 Ia 56 E. 8.2.1;Art. 462 ZGB"
```

etc.

**Notes:**

- Only citations exactly matching the strings in the retrieval corpus are considered correct.
- The order of citations within a row doesn't matter
- You can (and probably should to optimize above metric) provide a variable amount of citations.
- You can see the list of citations as the list of positives across the entire retrieval set.
- If you submit more different citations than there are in the gold set, there will definitely be some false positives in there



### Am I allowed to ...

Consider these criteria:

- Your solution should be **reproducible**, i.e. the competition host can recreate your results with reasonable time and money
- Your solution should be **scalable**, i.e. getting it to do inference on more samples should be somewhat economically viable (max. 10$ inference cost per sample)
- Your solution should **generalize**, i.e. be expected to perform well on samples beyond the test set from the same distribution

Examples:

1. Have domain expert annotating data manually? **No**, not scalable, not easily reproducible and if it's the test data they annotate it would not generalize.
2. Fine-tune a model? **Yes**, if done in a reproducible way and typically this is scalable and generalizes.
3. Scraping additional data? **Yes**, but please include or describe the code for this and upload the resulting data as Kaggle dataset so it can be reproducible. Careful with technical and legal limitations of this. Some websites can crash or block you when you scrape carelessly.

**Notes:**

- To verify generalization the competition host reserves the right to evaluate the notebook on a set of completely private queries that are **not** in the `test.csv`.
- Prizes are contingent on this re-evaluation + providing reproducible code for it.

If you are unsure about your solution meeting these criteria please reach out to the competition host ari.jordan@omnilex.ai.

### FAQ

See also the "Data FAQ" in the "Data" tab.

### What does “offline notebook” mean in practice?

Your submitted Kaggle notebook must reproduce submission.csv without internet access, in particular no external API calls. Any models/data you use must be available in the Kaggle environment (e.g., uploaded dataset, Kaggle packages, or preloaded model assets). Examples of allowed resources:

- Pretrained HF models uploaded as Kaggle datasets
- Sentence-transformers, rerankers, or small LLMs that run locally
- Custom indexes built inside the notebook
- `pip install` of public libraries is allowed.
  On uncertainty please ask.

### Do I need to understand Swiss law to compete?

No. You can treat citations as opaque labels and focus on retrieval/reranking. Legal intuition can still help though (query expansion, disambiguation, choosing how many citations to return).