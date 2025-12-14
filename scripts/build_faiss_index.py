#!/usr/bin/env python3
"""
build_faiss_index.py — Build FAISS index from embeddings
"""

import argparse
from pathlib import Path

import numpy as np
import faiss


def build_faiss_index(embeddings, index_type="IndexFlatIP"):
    """
    Build FAISS index from embeddings
    IndexFlatIP = Inner Product (for cosine similarity with normalized vectors)
    """
    dimension = embeddings.shape[1]
    
    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add vectors to index
    index.add(embeddings)
    
    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npy")
    parser.add_argument("--output", required=True, help="Output path for FAISS index")
    parser.add_argument("--index-type", default="IndexFlatIP", help="FAISS index type")
    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"Loaded embeddings with shape {embeddings.shape}")

    # Build index
    print(f"Building FAISS index (type: {args.index_type})...")
    index = build_faiss_index(embeddings, args.index_type)
    print(f"Index built with {index.ntotal} vectors")

    # Save index
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"Saved FAISS index to {output_path}")


if __name__ == "__main__":
    main()
