# rag_v2.py — Upgraded RAG, 100% local via Ollama
#   Step 2: Load from PDF / TXT files
#   Step 3: Chunking with overlap
#   Step 4: Persist embeddings to disk (ChromaDB)
#   Step 5: ChromaDB as vector store
#   Embeddings: nomic-embed-text via Ollama (no HuggingFace, no internet)
#
# Setup:
#   ollama pull llama3.2
#   ollama pull nomic-embed-text
#   pip install chromadb pypdf
#
# Usage:
#   mkdir docs        <- put your .txt or .pdf files here
#   python rag_v2.py

import os
import hashlib
import requests
from pathlib import Path
import chromadb

# ─── CONFIG ────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434"
GEN_MODEL       = "llama3.2"
EMBED_MODEL     = "nomic-embed-text"
DOCS_DIR        = "./docs"
CHROMA_DIR      = "./chroma_store"
COLLECTION      = "rag_knowledge"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 150
TOP_K           = 5
SCORE_THRESHOLD  = 0.40

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

# ─── STEP 2: DOCUMENT LOADERS ──────────────────────────────────────────────
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_documents(docs_dir: str) -> list[dict]:
    """Scan docs_dir and load all .txt and .pdf files."""
    docs = []
    for filepath in Path(docs_dir).glob("**/*"):
        if filepath.suffix.lower() == ".txt":
            text = load_txt(str(filepath))
        elif filepath.suffix.lower() == ".pdf":
            text = load_pdf(str(filepath))
        else:
            continue
        if text.strip():
            docs.append({"source": str(filepath), "text": text})
            print(f"  Loaded: {filepath.name} ({len(text)} chars)")
    return docs

# ─── STEP 3: CHUNKING WITH OVERLAP ─────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Slide a window of `chunk_size` chars across the text,
    stepping forward by (chunk_size - overlap) each time.

    Example with chunk_size=10, overlap=3:
      "ABCDEFGHIJKLMNO"
       chunk1: ABCDEFGHIJ
       chunk2: HIJKLMNO    <- starts 3 chars before chunk1 ended
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]  # drop tiny trailing fragments

def chunk_documents(docs: list[dict]) -> list[dict]:
    """Chunk all loaded documents, preserving source metadata."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{doc['source']}_{i}".encode()).hexdigest()
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "source": doc["source"],
                "chunk_index": i
            })
    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks

# ─── STEP 4 + 5: CHROMADB (PERSISTENT) ────────────────────────────────────
def get_or_create_collection():
    """
    PersistentClient saves everything to CHROMA_DIR on disk.
    First run  -> embeds and stores.
    Later runs -> loads instantly, skips re-embedding.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def index_chunks(collection, chunks: list[dict]):
    """Embed and store only chunks not already in ChromaDB."""
    existing_ids = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    if not new_chunks:
        print("  All chunks already indexed — skipping embedding.")
        return

    print(f"  Embedding {len(new_chunks)} new chunks via Ollama...")
    for i, chunk in enumerate(new_chunks):
        print(f"    [{i+1}/{len(new_chunks)}] {chunk['text'][:60]}...")
        embedding = ollama_embed([chunk["text"]])[0]
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            }]
        )
    print(f"  Done. {len(new_chunks)} chunks indexed.")

# ─── RETRIEVER ──────────────────────────────────────────────────────────────
def retrieve(query: str, collection) -> list[dict]:
    """Embed the query and find the top-K most similar chunks."""
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
        chunks.append({
            "text": text,
            "source": Path(meta["source"]).name,
            "score": round(1 - dist, 3)   # cosine distance -> similarity
        })
    return chunks

# ─── GENERATOR ──────────────────────────────────────────────────────────────
def ask(question: str, collection) -> str:
    retrieved = retrieve(question, collection)

    # Show retrieved chunks for transparency / debugging
    print("\n  [Retrieved]")
    for i, r in enumerate(retrieved):
        print(f"    {i+1}. score={r['score']} [{r['source']}] {r['text'][:80]}...")

    # Tight prompt — important for smaller models like LLaMA 3.2
    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
    )
    prompt = f"""Use only the context below to answer the question. Be concise and factual. \
If the answer is not in the context, say "I don't know".

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
    print("\n=== RAG v2 (Ollama-only) ===\n")

    # 1. Load documents
    print(f"[1/3] Loading documents from '{DOCS_DIR}'...")
    os.makedirs(DOCS_DIR, exist_ok=True)
    docs = load_documents(DOCS_DIR)
    if not docs:
        print(f"\n  No .txt or .pdf files found in '{DOCS_DIR}'.")
        print("  Add some files and rerun.\n")
        return

    # 2. Chunk
    print(f"\n[2/3] Chunking ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)...")
    chunks = chunk_documents(docs)

    # 3. Index into ChromaDB
    print(f"\n[3/3] Indexing into ChromaDB at '{CHROMA_DIR}'...")
    collection = get_or_create_collection()
    index_chunks(collection, chunks)

    # 4. Query loop
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
