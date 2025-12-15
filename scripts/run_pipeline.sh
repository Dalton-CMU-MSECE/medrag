#!/usr/bin/env bash
# run_pipeline.sh — Main script to run the complete RAG pipeline
# Usage: ./run_pipeline.sh [config_path] [run_id]

set -e

CONFIG_PATH="${1:-configs/pipeline_config.yaml}"
RUN_ID="${2:-$(date +%Y-%m-%dT%H:%M:%S)-run}"
RUN_DIR="runs/${RUN_ID}"

echo "==== Medical RAG Pipeline ===="
echo "Config: ${CONFIG_PATH}"
echo "Run ID: ${RUN_ID}"
echo "Run directory: ${RUN_DIR}"
echo ""

# Create run directory
mkdir -p "${RUN_DIR}"

# Step 1: Encode documents
echo "[1/3] Encoding documents..."
python scripts/encode_documents.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${RUN_DIR}"

# Step 2: Build FAISS index
echo "[2/3] Building FAISS index..."
python scripts/build_faiss_index.py \
  --embeddings "${RUN_DIR}/embeddings.npy" \
  --output "${RUN_DIR}/faiss.index"

# Step 3: Ingest into Elasticsearch
echo "[3/3] Ingesting into Elasticsearch..."
python scripts/ingest_elastic.py \
  --config "${CONFIG_PATH}" \
  --docs "data/processed/bioasq_round_1_docs.jsonl"

echo ""
echo "==== Pipeline complete ===="
echo "Artifacts saved to: ${RUN_DIR}"
echo "- embeddings.npy"
echo "- faiss.index"
echo "- embeddings_manifest.json"
echo "- run_manifest.json"
