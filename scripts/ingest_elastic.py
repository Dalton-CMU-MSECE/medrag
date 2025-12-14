#!/usr/bin/env python3
"""
ingest_elastic.py — Ingest documents into Elasticsearch for BM25 retrieval
"""

import argparse
import json

import yaml
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


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


def create_index(es, index_name):
    """Create Elasticsearch index with appropriate mappings"""
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists, deleting...")
        es.indices.delete(index=index_name)
    
    mappings = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "english"},
                "abstract": {"type": "text", "analyzer": "english"},
                "pub_date": {"type": "date"},
                "doi": {"type": "keyword"},
                "source": {"type": "keyword"}
            }
        }
    }
    
    es.indices.create(index=index_name, body=mappings)
    print(f"Created index '{index_name}'")


def generate_bulk_actions(documents, index_name):
    """Generate bulk actions for Elasticsearch"""
    for doc in documents:
        yield {
            "_index": index_name,
            "_id": doc["doc_id"],
            "_source": doc
        }


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Elasticsearch")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--docs", required=True, help="Path to docs.jsonl")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    bm25_config = config["bm25"]
    
    # Connect to Elasticsearch
    es_host = bm25_config["elasticsearch_host"]
    es_port = bm25_config["elasticsearch_port"]
    es = Elasticsearch([f"http://{es_host}:{es_port}"])
    
    print(f"Connected to Elasticsearch at {es_host}:{es_port}")
    
    # Create index
    index_name = bm25_config["index_name"]
    create_index(es, index_name)
    
    # Load documents
    print(f"Loading documents from {args.docs}...")
    documents = load_documents(args.docs)
    print(f"Loaded {len(documents)} documents")
    
    # Bulk ingest
    print(f"Ingesting documents into '{index_name}'...")
    success, failed = bulk(es, generate_bulk_actions(documents, index_name))
    print(f"Successfully ingested {success} documents")
    
    if failed:
        print(f"Failed to ingest {failed} documents")


if __name__ == "__main__":
    main()
