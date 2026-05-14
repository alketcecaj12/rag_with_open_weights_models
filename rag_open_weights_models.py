# rag_open_weights_models.py — Upgraded RAG, 100% local via Ollama
#
# Improvements over v1:
# - Page-level chunking: each chunk carries the exact PDF page it came from
# - Sentence-aware chunking: no definitions cut mid-sentence
# - Larger chunks (1000 chars) + wider overlap (150 chars)
# - TOP_K increased to 5 for better recall
# - Score threshold: low-quality chunks are filtered out before generation
# - Citations: model is instructed to cite page numbers in its answer
# - Retrieved chunks show page numbers in debug output
# - BM25 hybrid retrieval with Reciprocal Rank Fusion (RRF)
#
# NOTE on Ollama API endpoint:
#   Ollama >= 0.2 deprecated /api/embeddings in favour of /api/embed:
#     request key  "prompt"    -> "input"
#     response key "embedding" -> "embeddings"[0]
#
# Setup:
#   ollama pull llama3.2
#   ollama pull nomic-embed-text
#   pip install chromadb pypdf nltk rank_bm25
#   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
#
# If Ollama runs inside Docker, set OLLAMA_URL to one of:
#   http://localhost:11434              (Ollama port exposed on host)
#   http://host.docker.internal:11434  (script also inside a container)
#
# Usage:
#   mkdir docs   <- put your .txt or .pdf files here
#   python rag_open_weights_models.py

import os
import hashlib
import requests
from pathlib import Path

import chromadb
import nltk
from rank_bm25 import BM25Okapi

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OLLAMA_URL      = os.getenv("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL       = "llama3.2"
EMBED_MODEL     = "mxbai-embed-large"
DOCS_DIR        = "./docs"
CHROMA_DIR      = "./chroma_store"
COLLECTION      = "rag_knowledge"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 150
TOP_K           = 5
SCORE_THRESHOLD = 0.40
RRF_K           = 60

# ─── OLLAMA HELPERS ──────────────────────────────────────────────────────────
# Uses /api/embed (Ollama >= 0.2).
# Old /api/embeddings returns 404 on recent Ollama builds.

#What is Tier 1 capital?
#What is the definition of a Global Systemically Important Bank (G-SIB)?
#What does LCR stand for, and what is its minimum requirement?
#How does the Basel Framework define a trading book?
#What is the Net Stable Funding Ratio (NSFR)?

def ollama_embed(texts: list) -> list:
    embeddings = []
    for text in texts:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        response.raise_for_status()
        embeddings.append(response.json()["embeddings"][0])
    return embeddings


def ollama_generate(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"].strip()

# ─── DOCUMENT LOADERS ────────────────────────────────────────────────────────

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> list:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages


def load_documents(docs_dir: str) -> list:
    docs = []
    for filepath in Path(docs_dir).glob("**/*"):
        if filepath.suffix.lower() == ".txt":
            text = load_txt(str(filepath))
            if text.strip():
                docs.append({
                    "source": str(filepath),
                    "pages":  [{"text": text, "page": None}],
                })
                print(f"  Loaded: {filepath.name} ({len(text)} chars)")
        elif filepath.suffix.lower() == ".pdf":
            pages = load_pdf(str(filepath))
            if pages:
                total_chars = sum(len(p["text"]) for p in pages)
                docs.append({"source": str(filepath), "pages": pages})
                print(f"  Loaded: {filepath.name} ({len(pages)} pages, {total_chars} chars)")
    return docs

# ─── SENTENCE-AWARE CHUNKING ─────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    from nltk.tokenize import sent_tokenize
    sentences   = sent_tokenize(text)
    chunks      = []
    current     = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            kept, kept_len = [], 0
            for s in reversed(current):
                if kept_len + len(s) > overlap:
                    break
                kept.insert(0, s)
                kept_len += len(s)
            current, current_len = kept, kept_len
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) > 30]


def chunk_documents(docs: list) -> list:
    all_chunks = []
    for doc in docs:
        for page_block in doc["pages"]:
            chunks   = chunk_text(page_block["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            page_num = page_block["page"] or 0
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{doc['source']}_{page_num}_{i}".encode()
                ).hexdigest()
                all_chunks.append({
                    "id":          chunk_id,
                    "text":        chunk,
                    "source":      doc["source"],
                    "page":        page_num,
                    "chunk_index": i,
                })
    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks

# ─── CHROMADB ────────────────────────────────────────────────────────────────

def get_or_create_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(collection, chunks: list):
    existing_ids = set(collection.get()["ids"])
    new_chunks   = [c for c in chunks if c["id"] not in existing_ids]
    if not new_chunks:
        print("  All chunks already indexed — skipping embedding.")
        return
    print(f"  Embedding {len(new_chunks)} new chunks via Ollama...")
    for i, chunk in enumerate(new_chunks):
        print(f"  [{i+1}/{len(new_chunks)}] page={chunk['page']} {chunk['text'][:60]}...")
        embedding = ollama_embed([chunk["text"]])[0]
        collection.add(
            ids        = [chunk["id"]],
            embeddings = [embedding],
            documents  = [chunk["text"]],
            metadatas  = [{
                "source":      chunk["source"],
                "page":        chunk["page"],
                "chunk_index": chunk["chunk_index"],
            }],
        )
    print(f"  Done. {len(new_chunks)} chunks indexed.")

# ─── RETRIEVAL (hybrid: semantic + BM25 via RRF) ─────────────────────────────

class BM25Index:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.bm25   = BM25Okapi([c["text"].lower().split() for c in chunks])

    def search(self, query: str, top_k: int) -> list:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)[:top_k]
        return [
            {**chunk, "bm25_score": round(float(score), 4)}
            for score, chunk in ranked if score > 0
        ]


def retrieve(query: str, collection, bm25_index: BM25Index) -> list:
    # Semantic retrieval
    query_embedding = ollama_embed([query])[0]
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = TOP_K,
        include          = ["documents", "metadatas", "distances"],
    )
    semantic = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = round(1 - dist, 3)
        if score >= SCORE_THRESHOLD:
            semantic.append({
                "text":      text,
                "source":    Path(meta["source"]).name,
                "page":      meta.get("page", 0),
                "sem_score": score,
            })

    # BM25 retrieval
    bm25_hits = [
        {
            "text":       c["text"],
            "source":     Path(c["source"]).name,
            "page":       c["page"],
            "bm25_score": c["bm25_score"],
        }
        for c in bm25_index.search(query, TOP_K)
    ]

    # Reciprocal Rank Fusion
    rrf_scores, chunk_store = {}, {}
    for rank, chunk in enumerate(semantic):
        key = chunk["text"][:60]
        rrf_scores[key]  = rrf_scores.get(key, 0) + 1 / (rank + RRF_K)
        chunk_store[key] = chunk
    for rank, chunk in enumerate(bm25_hits):
        key = chunk["text"][:60]
        rrf_scores[key]  = rrf_scores.get(key, 0) + 1 / (rank + RRF_K)
        if key not in chunk_store:
            chunk_store[key] = chunk

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final  = []
    for key, rrf_score in ranked[:TOP_K]:
        chunk = chunk_store[key].copy()
        chunk["rrf_score"] = round(rrf_score, 6)
        final.append(chunk)
    return final

# ─── GENERATOR ───────────────────────────────────────────────────────────────

def ask(question: str, collection, bm25_index: BM25Index) -> str:
    retrieved = retrieve(question, collection, bm25_index)

    print("\n  [Retrieved]")
    if not retrieved:
        print("  No chunks above score threshold.")
    for i, r in enumerate(retrieved):
        page_label = f"p.{r['page']}" if r.get("page") else "n/a"
        print(f"  {i+1}. [{r['source']} {page_label}] {r['text'][:80]}...")

    if not retrieved:
        return (
            "I could not find relevant information in the indexed documents. "
            "Try rephrasing your question or check that the document has been indexed."
        )

    context = "\n\n".join(
        f"[Source: {r['source']}, Page: {r['page']}]\n{r['text']}"
        for r in retrieved
    )

    prompt = f"""Use only the context below to answer the question.
Be concise and factual.
If the answer is not in the context, say "I don't know".
At the end of your answer, always cite the source file and page number(s) \
where you found the information, like this:
Source: <filename>, Page <number>

Context:
{context}

Question: {question}

Answer:"""

    return ollama_generate(prompt)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n=== RAG (Ollama-only, with page citations) ===\n")

    print(f"[1/3] Loading documents from '{DOCS_DIR}'...")
    os.makedirs(DOCS_DIR, exist_ok=True)
    docs = load_documents(DOCS_DIR)
    if not docs:
        print(f"\n  No .txt or .pdf files found in '{DOCS_DIR}'.")
        print("  Add some files and rerun.\n")
        return

    print(f"\n[2/3] Chunking ({CHUNK_SIZE} chars max, {CHUNK_OVERLAP} overlap, sentence-aware)...")
    chunks = chunk_documents(docs)

    print(f"\n[3/3] Indexing into ChromaDB at '{CHROMA_DIR}'...")
    collection = get_or_create_collection()
    index_chunks(collection, chunks)
    bm25_index = BM25Index(chunks)

    print("\n=== Ready! Ask questions about your documents. ('quit' to exit) ===\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = ask(question, collection, bm25_index)
        print(f"\nRAG: {answer}\n")


if __name__ == "__main__":
    main()
