"""
Cross-encoder reranker for retrieved documents
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class CrossEncoderReranker:
    """Cross-encoder reranker using S-PubMedBERT-MS-MARCO"""
    
    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO", batch_size: int = 16):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"Warning: Could not load cross-encoder model: {e}")
            print("Using placeholder reranker")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_field: str = "abstract",
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Query text
            documents: List of documents with text field
            text_field: Field name containing document text
            top_k: Number of documents to return
        
        Returns:
            Reranked list of documents with updated scores
        """
        if self.model is None or not documents:
            # Return original documents if model not loaded
            return documents[:top_k]
        
        # Prepare query-document pairs
        pairs = [(query, doc.get(text_field, "")) for doc in documents]
        
        # Get reranking scores
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Add scores to documents
        reranked_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("score", 0.0)
            doc_copy["score"] = float(score)  # Update score
            reranked_docs.append(doc_copy)
        
        # Sort by rerank score
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_docs[:top_k]
    
    def score_pairs(self, query: str, texts: List[str]) -> np.ndarray:
        """
        Score query-text pairs
        
        Args:
            query: Query text
            texts: List of texts to score
        
        Returns:
            Array of scores
        """
        if self.model is None:
            # Return dummy scores
            return np.random.rand(len(texts))
        
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        return np.array(scores)
