"""
Hybrid retriever combining FAISS and BM25
"""

import numpy as np
from typing import List, Dict, Any, Tuple

from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    """Hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval"""
    
    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever
        
        Args:
            faiss_index: FAISS index for dense retrieval
            bm25_retriever: BM25 retriever for sparse retrieval
            alpha: Weight for combining scores (0-1, 1.0 = pure dense, 0.0 = pure sparse)
        """
        self.faiss_index = faiss_index
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
    
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k_dense: int = 100,
        top_k_sparse: int = 100,
        top_k_final: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            top_k_dense: Number of results from dense retrieval
            top_k_sparse: Number of results from sparse retrieval
            top_k_final: Number of final results
        
        Returns:
            List of retrieved documents with combined scores
        """
        # Dense retrieval (FAISS)
        dense_scores, dense_indices = self.faiss_index.search(query_embedding, top_k_dense)
        
        # Sparse retrieval (BM25)
        sparse_results = self.bm25_retriever.search(query, top_k_sparse)
        
        # Combine results
        combined_scores = {}
        
        # Add dense scores
        for idx, score in zip(dense_indices, dense_scores):
            # Prefer external doc_id mapping from FAISS if available
            try:
                if hasattr(self.faiss_index, "doc_ids") and self.faiss_index.doc_ids and int(idx) < len(self.faiss_index.doc_ids):
                    doc_id = str(self.faiss_index.doc_ids[int(idx)])
                else:
                    doc_id = str(int(idx))
            except Exception:
                doc_id = str(int(idx))
            combined_scores[doc_id] = {
                "dense_score": float(score),
                "sparse_score": 0.0,
                "index": int(idx)
            }
        
        # Add sparse scores
        for result in sparse_results:
            doc_id = result["doc_id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "dense_score": 0.0,
                    "sparse_score": result["score"],
                    "index": None
                }
            else:
                combined_scores[doc_id]["sparse_score"] = result["score"]
        
        # Normalize and combine scores
        # Normalize dense scores to 0-1
        dense_values = [v["dense_score"] for v in combined_scores.values()]
        if dense_values:
            max_dense = max(dense_values) if max(dense_values) > 0 else 1.0
            for doc_id in combined_scores:
                combined_scores[doc_id]["dense_score"] /= max_dense
        
        # Normalize sparse scores to 0-1
        sparse_values = [v["sparse_score"] for v in combined_scores.values()]
        if sparse_values:
            max_sparse = max(sparse_values) if max(sparse_values) > 0 else 1.0
            for doc_id in combined_scores:
                combined_scores[doc_id]["sparse_score"] /= max_sparse
            # Debug logging: show normalization factor and a few normalized scores
            try:
                print(f"BM25 normalization max_sparse={max_sparse:.4f}")
                sample = list(combined_scores.items())[:5]
                for did, val in sample:
                    print(f"doc_id={did} bm25_norm={val['sparse_score']:.4f} dense_norm={val['dense_score']:.4f}")
            except Exception:
                pass
        
        # Compute combined scores
        for doc_id in combined_scores:
            dense = combined_scores[doc_id]["dense_score"]
            sparse = combined_scores[doc_id]["sparse_score"]
            combined_scores[doc_id]["combined_score"] = self.alpha * dense + (1 - self.alpha) * sparse
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )[:top_k_final]

        # Debug logging: show top-5 combined scores
        try:
            head = sorted_results[:5]
            for did, s in head:
                print(
                    f"combined doc_id={did} combined={s['combined_score']:.4f} "
                    f"dense={s['dense_score']:.4f} sparse={s['sparse_score']:.4f}"
                )
        except Exception:
            pass
        
        # Format results
        results = []
        for doc_id, scores in sorted_results:
            results.append({
                "doc_id": doc_id,
                "score": scores["combined_score"],
                "dense_score": scores["dense_score"],
                "sparse_score": scores["sparse_score"],
                "index": scores["index"]
            })
        
        return results
