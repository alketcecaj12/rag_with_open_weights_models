# rag_open_weights_models.py — Upgraded RAG, 100% local via Ollama
#
#   Improvements over v1:
#   - Page-level chunking: each chunk carries the exact PDF page it came from
#   - Sentence-aware chunking: no definitions cut mid-sentence
#   - Larger chunks (1000 chars) + wider overlap (150 chars)
#   - TOP_K increased to 5 for better recall
#   - Score threshold: low-quality chunks are filtered out before generation
#   - Citations: model is instructed to cite page numbers in its answer
#   - Retrieved chunks show page numbers in debug output
#
# Setup:
#   ollama pull llama3.2
#   ollama pull nomic-embed-text
#   pip install chromadb pypdf nltk
#   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
#
# If Ollama runs inside Docker, set OLLAMA_URL to one of:
#   http://localhost:11434              (Ollama exposed on host port)
#   http://host.docker.internal:11434  (script also inside a container)
#
# Usage:
#   mkdir docs        <- put your .txt or .pdf files here
#   python rag_open_weights_models.py
#
# NOTE: If you previously ran the old version, delete ./chroma_store/ before
#       running this version — chunk IDs have changed.
#   rm -rf ./chroma_store

import os
import hashlib
import requests
from pathlib import Path
import chromadb
import nltk

# Download sentence tokenizer data (safe to run multiple times)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────
OLLAMA_URL       = "http://localhost:11434"   # change if Ollama is in Docker
GEN_MODEL        = "llama3.2"
EMBED_MODEL      = "nomic-embed-text"
DOCS_DIR         = "./docs"
CHROMA_DIR       = "./chroma_store"
COLLECTION       = "rag_knowledge"
CHUNK_SIZE       = 1000   # chars per chunk  (was 500)
CHUNK_OVERLAP    = 150    # overlap chars     (was 50)
TOP_K            = 5      # chunks retrieved  (was 3)
SCORE_THRESHOLD  = 0.40   # min cosine similarity to include a chunk

# ─── OLLAMA EMBEDDINGS ──────────────────────────────────────────────────────
def ollama_embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using nomic-embed-text via Ollama.
    Calls the local Ollama API — no internet required.
    """
    embeddings = []
    for text in texts:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text}
        )
        response.raise_for_status()
        embeddings.append(response.json()["embedding"])
    return embeddings

# ─── DOCUMENT LOADERS ──────────────────────────────────────────────────────
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path: str) -> list[dict]:
    """
    Load a PDF page by page.
    Returns a list of {"text": str, "page": int} dicts (1-indexed pages).
    Pages with no extractable text are skipped.
    """
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages

def load_documents(docs_dir: str) -> list[dict]:
    """
    Scan docs_dir for .txt and .pdf files.
    Returns a list of:
      {"source": str, "pages": [{"text": str, "page": int | None}, ...]}

    For .txt files, page is None (no page concept).
    For .pdf files, page is the 1-indexed page number.
    """
    docs = []
    for filepath in Path(docs_dir).glob("**/*"):
        if filepath.suffix.lower() == ".txt":
            text = load_txt(str(filepath))
            if text.strip():
                docs.append({
                    "source": str(filepath),
                    "pages":  [{"text": text, "page": None}]
                })
                print(f"  Loaded: {filepath.name} ({len(text)} chars, no page info)")

        elif filepath.suffix.lower() == ".pdf":
            pages = load_pdf(str(filepath))
            if pages:
                total_chars = sum(len(p["text"]) for p in pages)
                docs.append({
                    "source": str(filepath),
                    "pages":  pages
                })
                print(f"  Loaded: {filepath.name} ({len(pages)} pages, {total_chars} chars)")
    return docs

# ─── SENTENCE-AWARE CHUNKING ───────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks that respect sentence boundaries.

    Strategy:
      1. Tokenise into sentences using NLTK.
      2. Accumulate sentences until the chunk would exceed chunk_size.
      3. When a chunk is full, emit it and keep the last `overlap` chars
         worth of sentences as the start of the next chunk.

    This prevents definitions, rules, and numbered paragraphs from being
    cut mid-sentence, which was the main cause of "I don't know" answers
    on the Basel Framework.
    """
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks  = []
    current = []          # sentences in the current chunk
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # If adding this sentence would overflow AND we already have content,
        # emit the current chunk and start a new one with overlap.
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))

            # Roll back: keep trailing sentences until we've kept ~overlap chars
            kept, kept_len = [], 0
            for s in reversed(current):
                if kept_len + len(s) > overlap:
                    break
                kept.insert(0, s)
                kept_len += len(s)
            current     = kept
            current_len = kept_len

        current.append(sent)
        current_len += sent_len

    # Emit whatever remains
    if current:
        chunks.append(" ".join(current))

    # Drop fragments that are too short to be useful
    return [c for c in chunks if len(c) > 30]


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Chunk every page block in every document.
    Each chunk carries:
      - id          : stable MD5 hash (source + page + chunk index)
      - text        : the chunk text
      - source      : original file path
      - page        : PDF page number (int) or 0 for .txt files
      - chunk_index : position within this page's chunks
    """
    all_chunks = []
    for doc in docs:
        for page_block in doc["pages"]:
            chunks = chunk_text(page_block["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            page_num = page_block["page"] or 0   # ChromaDB metadata needs int

            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{doc['source']}_{page_num}_{i}".encode()
                ).hexdigest()
                all_chunks.append({
                    "id":          chunk_id,
                    "text":        chunk,
                    "source":      doc["source"],
                    "page":        page_num,
                    "chunk_index": i
                })

    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks

# ─── CHROMADB (PERSISTENT) ─────────────────────────────────────────────────
def get_or_create_collection():
    """
    PersistentClient saves everything to CHROMA_DIR on disk.
    First run  -> embeds and stores.
    Later runs -> loads instantly, skips re-embedding.
    """
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def index_chunks(collection, chunks: list[dict]):
    """Embed and store only chunks not already in ChromaDB."""
    existing_ids = set(collection.get()["ids"])
    new_chunks   = [c for c in chunks if c["id"] not in existing_ids]

    if not new_chunks:
        print("  All chunks already indexed — skipping embedding.")
        return

    print(f"  Embedding {len(new_chunks)} new chunks via Ollama...")
    for i, chunk in enumerate(new_chunks):
        print(f"    [{i+1}/{len(new_chunks)}] page={chunk['page']} {chunk['text'][:60]}...")
        embedding = ollama_embed([chunk["text"]])[0]
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "source":      chunk["source"],
                "page":        chunk["page"],        # int (0 for txt)
                "chunk_index": chunk["chunk_index"]
            }]
        )
    print(f"  Done. {len(new_chunks)} chunks indexed.")

# ─── RETRIEVER ──────────────────────────────────────────────────────────────
def retrieve(query: str, collection) -> list[dict]:
    """
    Embed the query, find top-K most similar chunks, and filter by
    SCORE_THRESHOLD so low-quality matches don't pollute the context.
    """
    query_embedding = ollama_embed([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = round(1 - dist, 3)   # cosine distance -> similarity
        if score >= SCORE_THRESHOLD:
            chunks.append({
                "text":   text,
                "source": Path(meta["source"]).name,
                "page":   meta.get("page", 0),
                "score":  score
            })

    return chunks

# ─── GENERATOR ──────────────────────────────────────────────────────────────
def ask(question: str, collection) -> str:
    retrieved = retrieve(question, collection)

    # Debug: show what was retrieved
    print("\n  [Retrieved]")
    if not retrieved:
        print("    No chunks above score threshold.")
    for i, r in enumerate(retrieved):
        page_label = f"p.{r['page']}" if r["page"] else "n/a"
        print(f"    {i+1}. score={r['score']} [{r['source']} {page_label}] "
              f"{r['text'][:80]}...")

    # Nothing good enough -> say so immediately
    if not retrieved:
        return (
            "I could not find relevant information in the indexed documents. "
            "Try rephrasing your question or check that the document has been indexed."
        )

    # Build context with explicit page references
    context = "\n\n".join(
        f"[Source: {r['source']}, Page: {r['page']}]\n{r['text']}"
        for r in retrieved
    )

    # The prompt instructs the model to cite page numbers
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

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"].strip()

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("\n=== RAG (Ollama-only, with page citations) ===\n")

    # 1. Load documents
    print(f"[1/3] Loading documents from '{DOCS_DIR}'...")
    os.makedirs(DOCS_DIR, exist_ok=True)
    docs = load_documents(DOCS_DIR)
    if not docs:
        print(f"\n  No .txt or .pdf files found in '{DOCS_DIR}'.")
        print("  Add some files and rerun.\n")
        return

    # 2. Chunk (sentence-aware, page-level)
    print(f"\n[2/3] Chunking ({CHUNK_SIZE} chars max, {CHUNK_OVERLAP} overlap, "
          f"sentence-aware)...")
    chunks = chunk_documents(docs)

    # 3. Index into ChromaDB
    print(f"\n[3/3] Indexing into ChromaDB at '{CHROMA_DIR}'...")
    collection = get_or_create_collection()
    index_chunks(collection, chunks)

    # 4. Interactive query loop
    print("\n=== Ready! Ask questions about your documents. ('quit' to exit) ===\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = ask(question, collection)
        print(f"\nRAG: {answer}\n")

if __name__ == "__main__":
    main()
