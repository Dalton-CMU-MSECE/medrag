"""
BM25 retriever using Elasticsearch
"""

from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class BM25Retriever:
    """BM25 lexical retriever using Elasticsearch"""
    
    def __init__(self, host: str = "localhost", port: int = 9200, index_name: str = "medical_docs"):
        """
        Initialize BM25 retriever
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            index_name: Name of the index
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        self.es = None
        self._connect()
    
    def _connect(self):
        """Connect to Elasticsearch"""
        try:
            self.es = Elasticsearch([f"http://{self.host}:{self.port}"])
            # Test connection
            if not self.es.ping():
                print(f"Warning: Could not connect to Elasticsearch at {self.host}:{self.port}")
                self.es = None
        except Exception as e:
            print(f"Warning: Elasticsearch connection failed: {e}")
            self.es = None
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of search results with doc_id and score
        """
        if self.es is None:
            return []
        
        # BM25 query on title and abstract fields
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "abstract"],
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                src = hit.get("_source", {}) or {}
                # Prefer external PMID from source metadata if available; fallback to ES _id
                try:
                    pmid = src.get("metadata", {}).get("pmid")
                except Exception:
                    pmid = None
                doc_id_val = pmid if pmid else hit.get("_id")
                item = {
                    "doc_id": str(doc_id_val),
                    "score": hit.get("_score", 0.0),
                    "source": src
                }
                # Debug logging: raw BM25 _score per hit
                try:
                    print(f"BM25 raw _score doc_id={item['doc_id']}: {item['score']}")
                except Exception:
                    pass
                results.append(item)
            
            return results
        
        except Exception as e:
            print(f"Elasticsearch search error: {e}")
            return []
    
    def index_exists(self) -> bool:
        """Check if the index exists"""
        if self.es is None:
            return False
        try:
            return self.es.indices.exists(index=self.index_name)
        except:
            return False

    def create_index(self):
        """Create index with basic BM25-friendly mappings if it doesn't exist"""
        if self.es is None:
            return False
        if self.index_exists():
            return True
        settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "english_custom": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text", "analyzer": "english"},
                    "abstract": {"type": "text", "analyzer": "english"},
                    "pub_date": {"type": "date", "ignore_malformed": True},
                    "metadata": {"type": "object"}
                }
            }
        }
        try:
            self.es.indices.create(index=self.index_name, body=settings)
            return True
        except Exception:
            # If race condition or already exists
            return self.index_exists()

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Bulk index documents into Elasticsearch"""
        if self.es is None or not documents:
            return False
        self.create_index()
        actions = []
        for doc in documents:
            doc_id = str(doc.get("doc_id"))
            source = {
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", ""),
                "pub_date": doc.get("pub_date"),
                "metadata": doc.get("metadata", {})
            }
            actions.append({
                "_index": self.index_name,
                "_id": doc_id,
                "_source": source
            })
        try:
            bulk(self.es, actions)
            return True
        except Exception as e:
            print(f"Elasticsearch bulk index error: {e}")
            return False
