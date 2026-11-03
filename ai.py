#!/usr/bin/env python3
"""
FTC GPT backend: serves the web UI and answers questions with RAG over Ollama.

Loads vectorized chunks (from your ingest script), retrieves the most similar
ones for each question, and has an Ollama model answer from them.

Usage:
    python ai.py [chunks_file] [--model NAME] [--port N]
    (opens http://localhost:8000 in your browser)

Environment: OLLAMA_HOST (default http://localhost:11434), FTC_MODEL (answer model)

Build an executable:
    pip install pyinstaller numpy requests
    pyinstaller --onefile --name ftc-gpt --add-data "ftc-gpt.html:." --add-data "chunks.jsonl:." ai.py
    (on Windows use ";" instead of ":" in --add-data)

Dependencies: numpy, requests
"""

import argparse
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import requests

# ============================================================================
# Configuration
# ============================================================================
def bundled(name: str) -> Path:
    """Path to a file packed inside the PyInstaller executable (or next to this script)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / name


def default_chunks() -> str:
    """Prefer a chunks.jsonl beside the executable/script (easy to update), else the bundled one."""
    here = Path(sys.executable if getattr(sys, "frozen", False) else __file__).parent
    return str(here / "chunks.jsonl" if (here / "chunks.jsonl").exists() else bundled("chunks.jsonl"))


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EMBED_MODEL = "nomic-embed-text"   # MUST match the model used at ingest time
DEFAULT_ANSWER_MODEL = os.environ.get("FTC_MODEL", "gemma4:e4b")
TOP_K = 18                         # Max chunks handed to the model
SIMILARITY_THRESHOLD = 0.2         # Minimum cosine similarity to include
NUM_CTX = 12288                      # Ollama context window: chunks + history + reply
HISTORY_TURNS = 6                  # Prior messages kept for follow-up questions
MAX_FILE_CHARS = 20000             # Per attached file
HTML_PATH = bundled("ftc-gpt.html")

SYSTEM_PROMPT = (
    "You are FTC GPT, a helpful assistant for FIRST Tech Challenge students, mentors "
    "and volunteers. Answer **concisely**, using only the information in the provided "
    "context. If the answer is not in the context, say so explicitly and suggest the "
    "official Q&A forum. Don't quote exact file names; use general terms like "
    "'in the Game Manual' or 'in the goBILDA parts list'."
)


# ============================================================================
# Chunk store: load once, search with a single matrix multiply
# ============================================================================
class ChunkStore:
    def __init__(self, filepath: str):
        path = Path(filepath)
        if not path.exists() or path.stat().st_size == 0:
            sys.exit(f"Error: '{filepath}' is missing or empty. Run your ingest script first.")

        chunks, vectors = [], []
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠ Line {n}: JSON parse error - {e}")
                    continue
                if rec.get("embedding") and "text" in rec and "source_file" in rec:
                    vectors.append(rec.pop("embedding"))
                    chunks.append(rec)
                else:
                    print(f"⚠ Line {n}: missing required fields, skipping")

        if not chunks:
            sys.exit("Error: no valid chunks loaded. Check the JSONL format.")
        if len({len(v) for v in vectors}) > 1:
            sys.exit("Error: chunks have inconsistent embedding dimensions.")

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.matrix = matrix / norms  # pre-normalized: dot product == cosine similarity
        self.chunks = chunks

    def search(self, query_embedding) -> list:
        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0 or q.shape[0] != self.matrix.shape[1]:
            return []
        sims = self.matrix @ (q / norm)
        top = np.argsort(-sims)[:TOP_K]
        return [{**self.chunks[i], "similarity": float(sims[i])}
                for i in top if sims[i] >= SIMILARITY_THRESHOLD]


# ============================================================================
# RAG: retrieve + answer
# ============================================================================
def ollama(path: str, payload: dict, timeout: int) -> dict:
    resp = requests.post(f"{OLLAMA_HOST}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def answer(messages: list, store: ChunkStore, model: str) -> dict:
    """messages: [{role, content, files?}, ...] with the new question last."""
    last = messages[-1]
    files = last.get("files") or []
    question = (last.get("content") or "").strip() or "Please review the attached file(s)."

    embedding = ollama("/api/embed", {"model": EMBED_MODEL, "input": [question]}, 30)["embeddings"][0]
    results = store.search(embedding)

    if not results and not files:
        return {"answer": "No relevant information found in the documents.", "sources": []}

    context = "\n\n---\n\n".join(
        f"[{r['source_file']} (chunk {r['chunk_index']})]\n{r['text']}" for r in results
    )
    attached = "\n\n".join(
        f"--- Attached: {f['name']} ---\n{f.get('text', '')[:MAX_FILE_CHARS]}" for f in files
    )
    user_prompt = f"Context from documents:\n{context}\n\n"
    if attached:
        user_prompt += f"{attached}\n\n"
    user_prompt += f"---\n\nQuestion: {question}"

    history = [{"role": m["role"], "content": m["content"]}
               for m in messages[:-1][-HISTORY_TURNS:]]
    chat = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
            {"role": "user", "content": user_prompt}]

    reply = ollama("/api/chat", {
        "model": model,
        "messages": chat,
        "stream": False,
        "options": {"num_ctx": NUM_CTX},
    }, 120)
    sources = list(dict.fromkeys(r["source_file"] for r in results))
    return {"answer": reply["message"]["content"], "sources": sources}


# ============================================================================
# HTTP server: serves the UI and the chat endpoint
# ============================================================================
class Handler(BaseHTTPRequestHandler):
    store: ChunkStore
    model: str

    def _send(self, code: int, body, ctype: str = "application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(200, {"model": self.model, "chunks": len(self.store.chunks)})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            messages = json.loads(self.rfile.read(length))["messages"]
            self._send(200, answer(messages, self.store, self.model))
        except requests.RequestException as e:
            self._send(502, {"error": f"Ollama request failed: {e}"})
        except (KeyError, IndexError, TypeError, ValueError):
            self._send(400, {"error": "Bad request"})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="FTC GPT RAG server")
    parser.add_argument("chunks_file", nargs="?", default=default_chunks())
    parser.add_argument("--model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not HTML_PATH.exists():
        sys.exit(f"Error: {HTML_PATH.name} not found.")

    Handler.store = ChunkStore(args.chunks_file)
    Handler.model = args.model
    print(f"Loaded {len(Handler.store.chunks)} chunks | model: {args.model}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"FTC GPT running at http://localhost:{args.port}  (Ctrl+C to stop)")
    webbrowser.open(f"http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFarewell :)")


if __name__ == "__main__":
    main()
