# RAG with Open-Weight Models — Local, 100% Offline

A fully local Retrieval-Augmented Generation (RAG) pipeline using
**Llama 3.2** for generation and **nomic-embed-text** for embeddings,
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
│  PDF / TXT file  ──►  pypdf loader  ──►  char-level chunker        │
│                        (500 chars,        (chunk_size=500,          │
│                         50 overlap)        overlap=50)              │
│                              │                                      │
│                              ▼                                      │
│                   nomic-embed-text (Ollama)                         │
│                    local embedding model                            │
│                              │                                      │
│                              ▼                                      │
│                   ChromaDB  (cosine similarity,                     │
│                    persisted to ./chroma_store)                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  QUERY  (runs on every question)                                    │
│                                                                     │
│  User question  ──►  nomic-embed-text  ──►  ChromaDB top-3 chunks  │
│                                                    │                │
│                                                    ▼                │
│                              Prompt = context + question            │
│                                                    │                │
│                                                    ▼                │
│                              llama3.2 (Ollama)  ──►  Answer        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key configuration values

| Parameter      | Value  | Effect                                           |
|----------------|--------|--------------------------------------------------|
| `CHUNK_SIZE`   | 500    | Characters per chunk (balance recall vs. noise)  |
| `CHUNK_OVERLAP`| 50     | Overlap prevents cutting sentences at boundaries |
| `TOP_K`        | 3      | How many chunks are injected into the prompt     |
| `GEN_MODEL`    | llama3.2 | 4-bit quantised generation model              |
| `EMBED_MODEL`  | nomic-embed-text | 4-bit quantised embedding model      |

---

## 3. Experiment — Setup & Running

### Prerequisites

```bash
# 1. Pull models inside your Ollama Docker container
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Install Python dependencies
pip install chromadb pypdf
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

Then update the constant at the top of `rag_open_weights_models.py`:

```python
OLLAMA_URL = "http://host.docker.internal:11434"   # <- whichever resolves
```

### First run (indexing)

```bash
mkdir docs
cp BaselFramework_.pdf docs/
python rag_open_weights_models.py
```

The first run embeds all chunks and writes them to `./chroma_store/`.
This is the slow step — a 1 982-page document will produce thousands of
chunks. Subsequent runs skip re-embedding and load instantly.

### Expected first-run output

```
=== RAG v2 (Ollama-only) ===

[1/3] Loading documents from './docs'...
  Loaded: BaselFramework_.pdf (XXXXXX chars)

[2/3] Chunking (500 chars, 50 overlap)...
  Total chunks: XXXX

[3/3] Indexing into ChromaDB at './chroma_store'...
  Embedding XXXX new chunks via Ollama...
  ...
  Done. XXXX chunks indexed.

=== Ready! Ask questions about your documents. ('quit' to exit) ===
You:
```

---

## 4. Analysis — Test Questions for the Basel Framework

The Basel Framework spans many regulatory domains. The questions below
are grouped by topic and by cognitive difficulty, so you can
systematically test retrieval quality from shallow factual recall to
deep multi-concept reasoning.

---

### Category A — Factual / Definition questions
> Test: Can the RAG retrieve a single precise definition?

```
What is Tier 1 capital?

What is the definition of a Global Systemically Important Bank (G-SIB)?

What does LCR stand for, and what is its minimum requirement?

How does the Basel Framework define a trading book?

What is the Net Stable Funding Ratio (NSFR)?

What is a credit conversion factor (CCF)?
```

---

### Category B — Numerical / Threshold questions
> Test: Can the RAG retrieve specific numbers buried in regulatory text?

```
What is the minimum Common Equity Tier 1 (CET1) ratio under Basel III?

What is the capital conservation buffer requirement?

What leverage ratio must G-SIBs maintain?

What haircut applies to Level 1 HQLA assets under the LCR?

What is the large exposure limit as a percentage of Tier 1 capital?
```

---

### Category C — Process / How-to questions
> Test: Can the RAG reconstruct a multi-step regulatory procedure?

```
How is Risk-Weighted Assets (RWA) calculated under the Standardised Approach for credit risk?

How does a bank calculate its Liquidity Coverage Ratio?

What are the steps to qualify for the IRB approach?

How is the leverage ratio exposure measure calculated?

How are securitisation exposures treated under the Standardised Approach?
```

---

### Category D — Comparative questions
> Test: Can the RAG surface distinctions between related concepts?

```
What is the difference between the Standardised Approach and the IRB approach for credit risk?

How does the banking book differ from the trading book?

What distinguishes Tier 1 capital from Tier 2 capital?

What is the difference between the LCR and the NSFR?

How does the treatment of G-SIBs differ from D-SIBs?
```

---

### Category E — Edge case / Out-of-scope questions
> Test: Does the RAG correctly say "I don't know" when the answer is absent?

```
What is the current ECB interest rate?

Who is the current Chair of the Federal Reserve?

What does IFRS 9 say about expected credit loss provisioning?
```

These questions are deliberately **outside the document**. A
well-functioning RAG should return *"I don't know"* rather than
hallucinate, because the prompt instructs: *"If the answer is not in
the context, say 'I don't know'."*

---

### Category F — Advanced / Multi-hop questions
> Test: Can the RAG chain concepts across retrieved chunks?

```
How does a bank's CET1 ratio affect its eligibility to pay dividends under the capital conservation buffer rules?

Under what conditions can a bank use internal models for market risk, and what backtesting requirements apply?

How are cryptoasset exposures classified and what capital treatment do they receive?

What supervisory review principles apply to interest rate risk in the banking book (IRRBB)?
```

---

## 5. Conclusion — What to look for when evaluating responses

| Signal | What it means |
|--------|---------------|
| Answer matches the document text | Retrieval and generation are working correctly |
| Score in `[Retrieved]` close to 1.0 | High cosine similarity — chunks are relevant |
| Score below ~0.5 | Weak retrieval — consider reducing `CHUNK_SIZE` or increasing `TOP_K` |
| "I don't know" on Category E questions | Grounding is working; model is not hallucinating |
| Answer on Category E questions | Hallucination — the model ignored the prompt constraint |
| Truncated or partial answers | Chunk boundaries cut the answer; try increasing `CHUNK_SIZE` |
| Wrong answer despite relevant chunks | Generation model size limitation; llama3.2 is small |

### Suggested tuning experiments

```python
# In rag_open_weights_models.py — try these combinations:

CHUNK_SIZE    = 800    # Larger chunks = more context per retrieval
CHUNK_OVERLAP = 100   # More overlap = fewer broken sentences
TOP_K         = 5     # More chunks = richer context, but longer prompts
```

---

## File structure after first run

```
project/
├── rag_open_weights_models.py   # Main script
├── docs/
│   └── BaselFramework_.pdf      # Source document (1982 pages)
├── chroma_store/                # Auto-created — persisted vector index
│   └── ...
└── README.md                    # This file
```

---

## Dependencies

| Package    | Purpose                          |
|------------|----------------------------------|
| `chromadb` | Vector store (local, persistent) |
| `pypdf`    | PDF text extraction              |
| `requests` | Ollama API calls                 |
| `ollama`   | llama3.2 + nomic-embed-text      |


--------------------------------------------------------

## Side loading the models from HuggingFace 

For nomic-embed-text, this typically requires a GGUF (GPT-Generated Unified Format) file. You can download the quantized GGUF version of Nomic Embed from reputable community repositories such as Hugging Face (huggingface.co). Utilizing a quantized format like Q4_K_M is often recommended for local deployment as it significantly reduces memory pressure while maintaining the model's 8192-token context window and high retrieval accuracy (Yadav et al., 6 Dec 2025).

Once the .gguf file is downloaded to your Windows host (e.g., in C:\Users\UserName\Downloads\nomic-embed-text.gguf), you must transfer it into the running Docker container. Use the docker cp command to move the file into a temporary directory within my_container. For example: docker cp C:\Users\UserName\Downloads\nomic-embed-text.gguf my_container:/tmp/nomic-embed-text.gguf. This ensures the weights are physically present within the container's isolated storage before the registration process begins.

## Creating the Local Modelfile
Ollama identifies and configures models through a manifest known as a Modelfile. To register your side-loaded weights, you must create a plain text file inside the container that points to the GGUF file you just uploaded. You can create this file using a simple redirected echo command: docker exec -it my_container sh -c "echo 'FROM /tmp/nomic-embed-text.gguf' > /tmp/Modelfile". This FROM instruction is the fundamental directive that binds the raw weights to a named model entity within the Ollama service.

For embedding models, it is often beneficial to specify additional parameters within the Modelfile to optimize performance. Although Nomic Embed is highly efficient, you can add lines such as PARAMETER num_ctx 8192 to explicitly define the context window length. This aligns the local runtime with the model's architectural capabilities, which have been shown to outperform proprietary long-context embedders on benchmarks like LoCo and MTEB (Nussbaum et al., 2024).

## Finalizing Registration and Verification
The final step is to invoke the Ollama binary to "create" the model from your Modelfile. Run the command: docker exec -it my_container ollama create nomic-embed-text -f /tmp/Modelfile. This command processes the GGUF file, generates the necessary metadata, and adds nomic-embed-text to the local library. Unlike the pull command, this operation is entirely offline and local to the container's filesystem, making it immune to the DNS and I/O timeout errors previously encountered.

## Plan B for Llama3.2  
If your image from which your container runs doesnt contain any model (Llama3.2 in this case), then you can use what is stated above to get a distilled Llama3.2 model from HuggingFace :-) and side-load it to your container.

## How to improve the results. 

- in general, longer chunks mean better understanding of the context
- while longer overlaps mean fewer broken sentences
- switch to sentence-aware chunking (structural fix)