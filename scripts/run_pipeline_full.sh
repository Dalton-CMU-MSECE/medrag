#!/usr/bin/env bash
# run_pipeline_full.sh — Full 2.4M pipeline runner (GPU-capable)
# Usage: ./scripts/run_pipeline_full.sh configs/pipeline_config_2.4M.yaml

set -euo pipefail

CONFIG_PATH="${1:-configs/pipeline_config_2.4M.yaml}"
RUN_DIR="runs/run_2"

echo "==== Medical RAG Full Pipeline (2.4M) ===="
echo "Config: ${CONFIG_PATH}"
echo "Run directory: ${RUN_DIR}"

mkdir -p "${RUN_DIR}"

# Extract docs path from YAML
DOCS_PATH=$(python - <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg['data']['docs_path'])
PY
"${CONFIG_PATH}")

# Step 1: Encode documents (GPU if encoder.device=cuda)
 echo "[1/3] Encoding documents to ${RUN_DIR}/embeddings.npy ..."
 python scripts/encode_documents.py \
   --config "${CONFIG_PATH}" \
   --output-dir "${RUN_DIR}"

# Step 2: Build FAISS index
 echo "[2/3] Building FAISS index ..."
 python scripts/build_faiss_index.py \
   --embeddings "${RUN_DIR}/embeddings.npy" \
   --output "${RUN_DIR}/faiss.index"

# Step 3: Ingest documents into Elasticsearch
 echo "[3/3] Ingesting ${DOCS_PATH} into Elasticsearch ..."
 python scripts/ingest_elastic.py \
   --config "${CONFIG_PATH}" \
   --docs "${DOCS_PATH}"

 echo "==== Full pipeline complete ===="
 echo "Artifacts saved to: ${RUN_DIR}"
 echo "- embeddings.npy"
 echo "- faiss.index"
 echo "- embeddings_manifest.json"
