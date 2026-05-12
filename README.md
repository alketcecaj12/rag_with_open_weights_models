# RAG with Open-Weight Models — Local, 100% Offline

A fully local Retrieval-Augmented Generation (RAG) pipeline using
**Llama 3.2** for generation and **mxbai-embed-large** for embeddings,
both running inside an **Ollama** Docker container. No OpenAI key, no
internet required at inference time.

---

## 1. Observation — What problem does this solve?

Large language models hallucinate when asked about documents they have
never seen. RAG fixes this by **retrieving the most relevant passages
from your own documents first**, then handing them as grounded context
to the model. This pipeline applies that pattern entirely locally:
no data ever leaves your machine.

The target document used for testing is the **BIS Basel Framework**
(`BaselFramework_.pdf`) — a 1 982-page technical regulatory compendium
published by the Basel Committee on Banking Supervision (BCBS). It
covers capital requirements, credit risk, market risk, liquidity ratios,
leverage, large exposures, operational risk, supervisory review, and
disclosure standards.

---

## 2. Hypothesis — How does the system work?

```
┌─────────────────────────────────────────────────────────────────────┐
│  INDEXING  (runs once, then cached)                                 │
│                                                                     │
│  PDF / TXT file  ──►  pypdf loader     ──►  sentence-aware chunker │
│                        (page-by-page)        (chunk_size=1000,      │
│                                               overlap=150)          │
│                              │                                      │
│                              ▼                                      │
│                   mxbai-embed-large (Ollama)                        │
│                    local embedding model                            │
│                              │                                      │
│                              ▼                                      │
│                   ChromaDB  (cosine similarity,                     │
│                    persisted to ./chroma_store)                     │
│                    metadata: source, page number, chunk index       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  QUERY  (runs on every question)                                    │
│                                                                     │
│  User question  ──►  mxbai-embed-large  ──►  ChromaDB top-5 chunks │
│                                              (semantic, score≥0.40) │
│                                +                                    │
│                              BM25 top-5 chunks (lexical)            │
│                                │                                    │
│                                ▼                                    │
│                    Reciprocal Rank Fusion (RRF)                     │
│                    → merged top-5 hybrid results                    │
│                                │                                    │
│                                ▼                                    │
│                    Prompt = context + question                      │
│                    (context includes page numbers)                  │
│                                │                                    │
│                                ▼                                    │
│                    llama3.2 (Ollama)  ──►  Answer + page citation   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key configuration values

| Parameter | Value | Effect |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap prevents cutting sentences at boundaries |
| `TOP_K` | 5 | Chunks injected into the prompt (both semantic and BM25) |
| `SCORE_THRESHOLD` | 0.40 | Minimum cosine similarity — low-quality chunks filtered out |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant |
| `GEN_MODEL` | llama3.2 | 4-bit quantised generation model |
| `EMBED_MODEL` | mxbai-embed-large | 4-bit quantised embedding model |

---

## 3. Experiment — Setup & Running

### Prerequisites

```bash
# 1. Pull models inside your Ollama Docker container
ollama pull llama3.2
ollama pull mxbai-embed-large

# 2. Install Python dependencies
pip install chromadb pypdf nltk rank_bm25 requests openpyxl

# 3. Download NLTK sentence tokenizer data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Docker / Ollama URL

Because Ollama runs inside a Docker container, `localhost` from a
script running **outside** the container will not resolve. Set the
correct host before running:

```bash
# Option A — Ollama exposed on host port 11434
export OLLAMA_URL="http://localhost:11434"

# Option B — Script also runs inside a container (use Docker bridge IP)
export OLLAMA_URL="http://host.docker.internal:11434"

# Option C — Look up the container IP directly
docker inspect <ollama_container_name> | grep '"IPAddress"'
export OLLAMA_URL="http://<container_ip>:11434"
```

### First run (indexing)

```bash
mkdir docs
cp BaselFramework_.pdf docs/
python rag_open_weights_models.py
```

The first run embeds all chunks and writes them to `./chroma_store/`.
Subsequent runs load the store instantly.

> **Note:** If you previously ran an older version with `nomic-embed-text`,
> delete `./chroma_store/` before switching to `mxbai-embed-large`:
> ```bash
> rm -rf ./chroma_store
> ```

---

## 4. Analysis — Test Questions for the Basel Framework

The questions below are grouped by topic and cognitive difficulty,
so you can systematically test retrieval quality from shallow factual
recall to deep multi-concept reasoning.

### Category A — Factual / Definition

```
What is Tier 1 capital?
What is the definition of a Global Systemically Important Bank (G-SIB)?
What does LCR stand for, and what is its minimum requirement?
How does the Basel Framework define a trading book?
What is the Net Stable Funding Ratio (NSFR)?
```

### Category B — Numerical / Threshold

```
What is the minimum Common Equity Tier 1 (CET1) ratio under Basel III?
What is the capital conservation buffer requirement?
What leverage ratio must G-SIBs maintain?
What is the large exposure limit as a percentage of Tier 1 capital?
```

### Category C — Process / How-to

```
How does a bank calculate its Liquidity Coverage Ratio?
How is Risk-Weighted Assets (RWA) calculated under the Standardised Approach?
How are securitisation exposures treated under the Standardised Approach?
How is the leverage ratio exposure measure calculated?
```

### Category D — Comparative

```
What is the difference between the Standardised Approach and the IRB approach for credit risk?
What distinguishes Tier 1 capital from Tier 2 capital?
What is the difference between the LCR and the NSFR?
```

### Category E — Out-of-scope (grounding test)

```
What is the current ECB interest rate?
Who is the current Chair of the Federal Reserve?
What does IFRS 9 say about expected credit loss provisioning?
```

These are deliberately **outside the document**. A well-functioning RAG
should return "I don't know" rather than hallucinate.

### Category F — Advanced / Multi-hop

```
How does a bank's CET1 ratio affect its eligibility to pay dividends
under the capital conservation buffer rules?

How are cryptoasset exposures classified and what capital treatment
do they receive?

What supervisory review principles apply to interest rate risk
in the banking book (IRRBB)?
```

---

## 5. Conclusion — Evaluation Frameworks

This project supports **two evaluation frameworks**, both running
fully locally via Ollama with no external API key required.
They use the same ground truth dataset so their scores can be
compared directly.

---

### Framework 1 — RAGAS (`rag_evaluate.py`)

RAGAS evaluates the RAG pipeline along two axes: retrieval quality
and generation quality.

#### Install

```bash
pip install ragas chromadb pypdf nltk rank_bm25 openpyxl requests
```

#### Run

```bash
python rag_evaluate.py
# Output: rag_eval_report_<timestamp>.xlsx
```

#### Metrics

| Metric | Layer | What it measures |
|---|---|---|
| Context Precision | Retriever | Of the retrieved chunks, what fraction were relevant? (PRECISION) |
| Context Recall | Retriever | How much of the correct answer was present in the chunks? (RECALL) |
| Faithfulness | Generator | Are the answer's claims grounded in the retrieved context? |
| Answer Relevancy | Generator | Does the answer actually address the question? |
| Page Accuracy | Retriever | Did the retriever find a chunk from the correct page? (deterministic) |

RAGAS uses free-form LLM prompts and parses a numeric score from the
response. It does **not** produce a natural-language reason for each score.

---

### Framework 2 — DeepEval (`rag_evaluate_deepeval.py`)

DeepEval uses Pydantic-validated structured LLM-judge calls, making
scores more deterministic than RAGAS. Its key unique feature is a
**natural-language reason** for every score — useful for debugging
specific retrieval or generation failures.

#### Install

```bash
pip install deepeval chromadb pypdf nltk rank_bm25 openpyxl requests

# Point DeepEval at your local Ollama (no OpenAI key needed)
deepeval set-ollama --model=llama3.2 --base-url="http://localhost:11434"
```

#### Run

```bash
python rag_evaluate_deepeval.py
# Output: deepeval_report_<timestamp>.xlsx
```

#### Metrics

| Metric | Layer | What it measures | Direction |
|---|---|---|---|
| FaithfulnessMetric | Generator | Does the answer contradict the retrieved context? | ↑ higher = better |
| AnswerRelevancyMetric | Generator | Does the answer address the question? | ↑ higher = better |
| ContextualPrecisionMetric | Retriever | Are relevant chunks ranked higher than noise? | ↑ higher = better |
| ContextualRecallMetric | Retriever | Does the context cover the expected answer? | ↑ higher = better |
| HallucinationMetric | End-to-end | What fraction of claims are unsupported by context? | ↓ **lower = better** |
| Page Accuracy | Retriever | Deterministic page-hit check (same as RAGAS version) | ↑ higher = better |

#### Report sheets

The DeepEval Excel report has an extra **"Reasons" sheet** not present
in the RAGAS report. It contains the LLM judge's plain-English
explanation for every score on every question — the primary tool for
diagnosing failures.

---

### Framework comparison

| | RAGAS | DeepEval |
|---|---|---|
| Judge mechanism | Free-text LLM prompt → parse number | Pydantic JSON → typed score |
| Reason per score | ❌ | ✅ (Reasons sheet) |
| Hallucination metric | Faithfulness (proxy) | Dedicated `HallucinationMetric` |
| Contextual Precision | Relevance count / total retrieved | Ranking-aware (penalises noise above signal) |
| Requires OpenAI | No (Ollama judge) | No (Ollama judge via `OllamaModel`) |
| Output | `rag_eval_report_*.xlsx` | `deepeval_report_*.xlsx` |

Both scripts use the **same 16-question ground truth dataset** and the
same hybrid RAG pipeline, so scores are directly comparable.

---

### Score interpretation guide

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.75 | Good | System is performing well on this metric |
| 0.40–0.74 | Fair | Acceptable but investigate individual failures |
| < 0.40 | Poor | Systematic issue — check `CHUNK_SIZE`, `TOP_K`, or embedding model |

> For `HallucinationMetric` (DeepEval only), invert this:
> score < 0.25 is good, score > 0.50 is a systematic problem.

---

### Suggested tuning experiments

```python
# In rag_open_weights_models.py — try these combinations:
CHUNK_SIZE      = 1200  # Larger chunks for dense regulatory text
CHUNK_OVERLAP   = 200   # More overlap = fewer broken definitions
TOP_K           = 7     # More chunks = richer context
SCORE_THRESHOLD = 0.35  # Lower if too many "not found" responses
```

---

## File structure

```
project/
├── rag_open_weights_models.py     # Main RAG script
├── rag_evaluate.py                # Evaluation: RAGAS metrics
├── rag_evaluate_deepeval.py       # Evaluation: DeepEval metrics
├── docs/
│   └── BaselFramework_.pdf        # Source document (1982 pages)
├── chroma_store/                  # Auto-created — persisted vector index
└── README.md                      # This file
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `chromadb` | Vector store (local, persistent) |
| `pypdf` | PDF text extraction (page-by-page) |
| `nltk` | Sentence tokenizer for sentence-aware chunks |
| `rank_bm25` | BM25 lexical retrieval for hybrid search |
| `requests` | Ollama API calls |
| `openpyxl` | Excel report generation |
| `ragas` | RAGAS evaluation framework |
| `deepeval` | DeepEval evaluation framework |
| `ollama` | llama3.2 + mxbai-embed-large |

---

## Side-loading models from HuggingFace

If your Ollama container has no internet access, download GGUF weights
from HuggingFace on another machine and transfer them via USB.

```bash
# Copy the GGUF file into the running container
docker cp /path/to/model.gguf my_container:/tmp/model.gguf

# Create a Modelfile inside the container
docker exec -it my_container sh -c "echo 'FROM /tmp/model.gguf' > /tmp/Modelfile"

# Register with Ollama
docker exec -it my_container ollama create my-model -f /tmp/Modelfile

# Verify
docker exec -it my_container ollama list
```

Recommended quantisation: `Q4_K_M` — good balance of accuracy and
memory footprint for both `llama3.2` and `mxbai-embed-large`.
