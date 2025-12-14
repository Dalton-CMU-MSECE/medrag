#!/usr/bin/env python3
"""
encode_documents.py — Encode documents using MedCPT encoder
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
from tqdm import tqdm


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


def load_documents(docs_path):
    """Load documents from JSONL file"""
    documents = []
    with open(docs_path, "r") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def encode_documents(documents, config):
    """
    Encode documents using the configured encoder
    In a real implementation, this would load the MedCPT model
    """
    # Placeholder: Generate random embeddings for demonstration
    embedding_dim = config["encoder"]["embedding_dim"]
    embeddings = np.random.randn(len(documents), embedding_dim).astype(np.float32)
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Encode documents for RAG pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory for embeddings")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Load documents
    docs_path = config["data"]["docs_path"]
    print(f"Loading documents from {docs_path}...")
    documents = load_documents(docs_path)
    print(f"Loaded {len(documents)} documents")

    # Encode documents
    print(f"Encoding documents with {config['encoder']['model']}...")
    embeddings = encode_documents(documents, config)
    print(f"Generated embeddings with shape {embeddings.shape}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Create manifest
    manifest = {
        "git_sha": get_git_sha(),
        "encoder": config["encoder"]["model"],
        "embedding_dim": config["encoder"]["embedding_dim"],
        "num_documents": len(documents),
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
