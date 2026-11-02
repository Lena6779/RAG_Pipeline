import chromadb
import requests
import json
import os
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"
DOCS_DIR = "docs"

# --- Step 1: Ingestion ---

def load_documents(directory):
    """Load all .txt and .md files from a directory and chunk them by paragraph."""
    docs = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.txt', '.md')):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                content = f.read()
            # Split on blank lines; skip empty pieces and markdown headings
            paragraphs = [
                p.strip() for p in content.split('\n\n')
                if p.strip() and not p.strip().startswith('#')
            ]
            for i, para in enumerate(paragraphs):
                docs.append({
                    "text": para,
                    "id": f"{filename}_{i}",
                    "metadata": {"source": filename, "chunk_index": str(i)}
                })
    return docs


def ingest(collection, docs_directory):
    """Load, chunk, and store documents in ChromaDB."""
    chunks = load_documents(docs_directory)
    if not chunks:
        print("No documents found!")
        return 0

    collection.upsert(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks]
    )
    return len(chunks)

# --- Step 2: Retrieval ---

def retrieve(collection, query, n_results=3):
    """Search ChromaDB and return the top matching chunks."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())
    )

    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })
    return chunks

# --- Step 3: Prompt Building ---

SYSTEM_PROMPT = """You are a helpful AI assistant for students learning AI engineering.
Answer the user's question based ONLY on the context provided below.

Rules:
- Only use information from the CONTEXT section
- If the context doesn't contain the answer, say "I don't have enough information
  to answer that based on the available documents."
- Cite your sources by mentioning the document name in parentheses, like (rag.md)
- Keep your response under 200 words
"""


def build_messages(question, retrieved_chunks):
    """Build the RAG prompt: system rules + retrieved context + the question."""
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk['metadata']['source']
        context_parts.append(f"[Source: {source}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    system_with_context = f"{SYSTEM_PROMPT}\nCONTEXT:\n{context}"

    return [
        {"role": "system", "content": system_with_context},
        {"role": "user", "content": question}
    ]

# --- Step 4: Generation ---

def generate(messages, stream=False):
    """Call Ollama and return the response text."""
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": MODEL,
            "messages": messages,
            "stream": stream
        }, stream=stream)

        # Ollama is running but something else went wrong (e.g. model not downloaded)
        if response.status_code != 200:
            return f"ERROR: Ollama returned status {response.status_code}. Have you run 'ollama pull {MODEL}'?"

        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full_response += token
            print()
            return full_response
        else:
            return response.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        message = "ERROR: Cannot connect to Ollama. Is it running? (ollama serve)"
        if stream:
            print(message)
        return message

# --- Step 5: The RAG Function ---

def rag_query(collection, question, n_results=3, stream=True):
    """Complete RAG pipeline: retrieve + build prompt + generate."""
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print(f"{'=' * 60}")

    # Retrieve and show chunks BEFORE the answer
    chunks = retrieve(collection, question, n_results)
    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk['text'][:80] + "..." if len(chunk['text']) > 80 else chunk['text']
        print(f"  [{i}] {chunk['metadata']['source']} (distance: {chunk['distance']:.4f})")
        print(f"      {preview}")

    # Build prompt and generate
    messages = build_messages(question, chunks)

    print("\nAnswer:\n")
    start = time.time()
    answer = generate(messages, stream=stream)
    elapsed = time.time() - start
    print(f"\n[Generated in {elapsed:.1f}s]")

    return answer

# --- Main ---

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./rag_db")
    collection = client.get_or_create_collection(
        "course_docs", configuration={"hnsw": {"space": "cosine"}}
    )

    if collection.count() == 0:
        count = ingest(collection, DOCS_DIR)
        print(f"Ingested {count} document chunks.")
    else:
        print(f"Using existing collection: {collection.count()} chunks")

    print("\nRAG Pipeline Ready! Ask questions about your documents.")
    print("Type 'quit' to exit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if question:
            rag_query(collection, question)