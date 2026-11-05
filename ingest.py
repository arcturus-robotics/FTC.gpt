"""
Usage:
    python ingest.py <input_dir> [output_file] [model (defaults to nomic-embed-text)]

Example:
    python ingest.py ./knowledge chunks.jsonl nomic-embed-text

Requires: requests, running Ollama server (default: http://yourURL:11434)
"""

import os
import sys
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

try:
    import requests
except ImportError:
    print("Error: 'requests' library required. Install with: pip install requests")
    sys.exit(1)

# Configuration

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_OUT = "chunks.jsonl"
CHUNK_SIZE = 256          # Target characters per chunk
CHUNK_OVERLAP = 32        # Overlap between chunks (characters)
PARAGRAPH_SEP = "\n\n"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf", ".docx"}

# Text Splitting
"""
Split text into overlapping chunks.
1. Try to split on paragraph boundaries
2. Fall back to sentence boundaries
3. Fall back to character boundaries with overlap
"""

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or len(text.strip()) == 0:
        return []

    # First pass: split into paragraphs
    paragraphs = re.split(PARAGRAPH_SEP, text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        test_chunk = f"{current_chunk}{PARAGRAPH_SEP}{para}" if current_chunk else para

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If paragraph itself is too long, split it further
            if len(para) > chunk_size:
                sub_chunks = _split_long_text(para, chunk_size, overlap)
                chunks.extend(sub_chunks[:-1])  # All but last (will merge with next)
                current_chunk = sub_chunks[-1]
            else:
                current_chunk = para

    # Save remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a long paragraph by sentences, then characters as last resort."""
    # Try splitting by sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sent in sentences:
        test = f"{current} {sent}".strip() if current else sent
        if len(test) <= chunk_size:
            current = test
        else:
            if current:
                chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    # If chunk is still too long, do character-level splitting with overlap
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = min(start + chunk_size, len(chunk))
                final_chunks.append(chunk[start:end])
                start += (chunk_size - overlap)

    return final_chunks


# Ollama Embedding

class OllamaEmbedder:

    def __init__(self, host: str = OLLAMA_HOST, model: str = DEFAULT_MODEL):
        self.host = host.rstrip('/')
        self.model = model
        self._verify_connection()

    def _safe_json(self, response, context="response"):
        """Parse JSON safely with helpful error messages."""
        try:
            return response.json()
        except json.JSONDecodeError as e:
            print(f"Failed to parse {context}: {e}")
            print(f"  Response text (first 500 chars): {response.text[:500]}")
            raise

    def _verify_connection(self):
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            data = self._safe_json(resp, "Ollama tags response")
            models = [m['name'] for m in data.get('models', [])]
            print(f"Connected to Ollama at {self.host}")
            if self.model not in models:
                # Try checking with version prefix removed
                base_name = self.model.split(':')[0]
                matching = [m for m in models if m.startswith(base_name)]
                if matching:
                    print(f"  Using available model variant: {matching[0]}")
                else:
                    print(f"⚠ Model '{self.model}' not found. Available: {', '.join(models)}")
        except requests.exceptions.ConnectionError as e:
            print(f"Cannot connect to Ollama at {self.host}: {e}")
            print("  Is Ollama running? Try: 'ollama serve'")
            sys.exit(1)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with robust parsing."""
        try:
            resp = requests.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=120
            )
            
            if resp.status_code != 200:
                print(f"Embedding API error (status {resp.status_code}):")
                print(f"  Response: {resp.text[:500]}")
                return [None] * len(texts)

            data = self._safe_json(resp, "embed response")
            
            # Handle different Ollama response formats
            if 'embedding' in data and isinstance(data['embedding'], list):
                # Newer format: single embedding or list of embeddings
                return [data['embedding']] if len(texts) == 1 else data['embedding']
            elif 'embeddings' in data:
                # Older format
                return data['embeddings']
            else:
                print(f"Unexpected embed response structure: {list(data.keys())}")
                return [None] * len(texts)

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [None] * len(texts)

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embed texts in batches for efficiency."""
        all_embeddings = []
        total = len(texts)

        print(f"Generating {total} embeddings...")
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed(batch)
            all_embeddings.extend(embeddings)

            done = min(i + batch_size, total)
            print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

        return all_embeddings


def extract_text(filepath: str) -> Optional[str]:
    """Extract text from a file, supporting plain text and PDFs."""
    ext = Path(filepath).suffix.lower()

    if ext == '.pdf':
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text
        except ImportError:
            print("Error: 'pypdf' library required for PDF support. Install with: pip install pypdf")
            sys.exit(1)
        except Exception as e:
            print(f"Error extracting PDF {filepath}: {e}")
            return None
    elif ext == '.docx':
        try:
            import docx
            document = docx.Document(filepath)
            parts = [p.text for p in document.paragraphs if p.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text.strip() for cell in row.cells))
            return "\n\n".join(parts)
        except ImportError:
            print("Error: 'python-docx' library required for DOCX support. Install with: pip install python-docx")
            sys.exit(1)
        except Exception as e:
            print(f"Error extracting DOCX {filepath}: {e}")
            return None
    else:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Could not read {filepath} as text: {e}")
            return None

# Document Processing Pipeline

def find_text_files(directory: str) -> List[str]:
    """Find all text files in directory (recursive)."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in sorted(filenames):
            ext = Path(filename).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, directory)
                files.append((filepath, rel_path))
    return files


def process_documents(input_dir: str, output_file: str, model: str):
    """Main processing pipeline."""
    print(f"Input directory: {input_dir}")
    print(f"Output file:     {output_file}")
    print(f"Embedding model: {model}")

    # 1. Find documents
    files = find_text_files(input_dir)
    if not files:
        print("No text files found.")
        return

    print(f"\nFound {len(files)} document(s):")
    for _, rel in files:
        print(f"  - {rel}")

    # 2. Split into chunks
    all_chunks = []  # List of dicts with metadata + text
    total_chars = 0

    for filepath, rel_path in files:
        content = extract_text(filepath)
        if content is None:
            print(f"⚠ Could not read {rel_path}")
            continue

        chunks = split_text(content)
        doc_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc_hash}_{idx}",
                "source_file": rel_path,
                "chunk_index": idx,
                "text": chunk,
                "char_count": len(chunk)
            })

        total_chars += len(content)
        print(f"  {rel_path}: {len(chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks from {total_chars:,} characters")

    if not all_chunks:
        return

    # Show sample chunk sizes
    sizes = [c['char_count'] for c in all_chunks]
    print(f"Chunk size stats: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

    # 3. Generate embeddings
    embedder = OllamaEmbedder(model=model)
    texts = [c['text'] for c in all_chunks]
    embeddings = embedder.embed_batch(texts)

    # Check if any embeddings succeeded
    if not any(embeddings):
        print("Failed to generate any embeddings. Check your model and connection.")
        return

    # 4. Save results as JSONL (one complete JSON object per line)
    print(f"\nSaving {len(all_chunks)} chunks to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk, emb in zip(all_chunks, embeddings):
            if emb is None:
                continue
                
            # Build record and ensure proper serialization
            record = {**chunk, "embedding": emb}
            
            try:
                json_line = json.dumps(record, ensure_ascii=False)
                f.write(json_line + '\n')
            except (TypeError, ValueError) as e:
                print(f"⚠ Could not serialize chunk {chunk['id']}: {e}")

    # Verify output file by re-reading and parsing each line
    success_count = 0
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line)
                    success_count += 1
                except json.JSONDecodeError:
                    print(f"Verification failed for a line")

    # Summary
    dim = len(embeddings[0]) if embeddings and embeddings[0] else 0
    print(f"\n✓ Done! {success_count}/{len(all_chunks)} chunks embedded ({dim} dimensions)")
    print(f"Output: {output_file}")

# Entry Point

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)

    process_documents(input_dir, output_file, model)