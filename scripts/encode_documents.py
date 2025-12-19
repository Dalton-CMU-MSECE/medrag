#!/usr/bin/env python3
"""
encode_documents.py — Batch-encode documents using configured encoder (GPU-ready)
Reads docs.jsonl and generates embeddings.npy and embeddings_manifest.json
"""

import argparse
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

# Optional: use actual encoders when available
try:
    from src.encoder.medcpt_encoder import MedCPTEncoder
    from src.encoder.biobert_encoder import BioBERTEncoder
except Exception:
    MedCPTEncoder = None
    BioBERTEncoder = None


def get_git_sha():
    """Get current git commit SHA"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def compute_file_sha256(filepath):
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def count_lines(docs_path: str) -> int:
    """Count non-empty lines in a JSONL file"""
    n = 0
    with open(docs_path, "r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def stream_documents(docs_path):
    """Yield documents from JSONL file one by one"""
    with open(docs_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_encoder(config):
    enc_cfg = config.get("encoder", {})
    backend = enc_cfg.get("backend", "medcpt").lower()
    model_name = enc_cfg.get("model", "ncbi/MedCPT-Query-Encoder")
    device = enc_cfg.get("device", "auto")
    if backend == "biobert" and BioBERTEncoder is not None:
        return BioBERTEncoder(model_name=model_name, device=device)
    if MedCPTEncoder is not None:
        return MedCPTEncoder(model_name=model_name, device=device)
    # Fallback: no encoders available
    return None

def batch_encode_to_memmap(docs_path: str, config: dict, output_path: Path) -> int:
    """Encode documents in batches directly into a .npy memmap file.
    Returns the number of documents processed.
    """
    embedding_dim = int(config["encoder"]["embedding_dim"])
    batch_size = int(config["encoder"].get("batch_size", 32))
    total = count_lines(docs_path)
    if total == 0:
        # Create empty file
        mm = open_memmap(str(output_path), mode="w+", dtype=np.float32, shape=(0, embedding_dim))
        del mm
        return 0

    encoder = build_encoder(config)
    mm = open_memmap(str(output_path), mode="w+", dtype=np.float32, shape=(total, embedding_dim))

    def _encode_texts(texts: list[str]) -> np.ndarray:
        if encoder is None:
            # Placeholder: random embeddings
            emb = np.random.randn(len(texts), embedding_dim).astype(np.float32)
        else:
            emb = encoder.encode(texts, batch_size=batch_size, normalize=True)
        # Ensure float32 and normalized (defensive)
        emb = emb.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-8)
        return emb

    i = 0
    buf_ids = []
    buf_texts = []
    for doc in tqdm(stream_documents(docs_path), total=total, desc="Encoding"):
        # Prefer abstract; fallback to text/title
        abstract = (doc.get("abstract") or doc.get("text") or "").strip()
        if not abstract:
            # write zeros for empty abstracts to keep alignment
            buf_ids.append(doc.get("doc_id"))
            buf_texts.append("")
        else:
            buf_ids.append(doc.get("doc_id"))
            buf_texts.append(abstract)

        # Flush batch
        if len(buf_texts) >= batch_size:
            emb = _encode_texts(buf_texts)
            mm[i:i+len(emb), :] = emb
            i += len(emb)
            buf_ids.clear()
            buf_texts.clear()

    # Flush tail
    if buf_texts:
        emb = _encode_texts(buf_texts)
        mm[i:i+len(emb), :] = emb
        i += len(emb)

    del mm  # flush to disk
    return i


def main():
    parser = argparse.ArgumentParser(description="Encode documents for RAG pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory for embeddings")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Source docs path
    docs_path = config["data"]["docs_path"]
    print(f"Preparing to encode documents from {docs_path}...")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode directly to .npy memmap on disk (GPU-ready, batched)
    embeddings_path = output_dir / "embeddings.npy"
    print(f"Encoding with {config['encoder']['model']} (device={config['encoder'].get('device','auto')})...")
    n_docs = batch_encode_to_memmap(docs_path, config, embeddings_path)
    print(f"Saved embeddings to {embeddings_path} (n_docs={n_docs})")

    # Create manifest
    manifest = {
        "git_sha": get_git_sha(),
        "encoder": config["encoder"]["model"],
        "embedding_dim": config["encoder"]["embedding_dim"],
        "num_documents": n_docs,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_sha256": compute_file_sha256(docs_path),
        "embeddings_sha256": compute_file_sha256(embeddings_path)
    }

    manifest_path = output_dir / "embeddings_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
