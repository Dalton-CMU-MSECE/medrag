"""
Medical RAG Pipeline - Main orchestration
"""

import os
from datetime import datetime
from typing import Dict, Any, List
import uuid

from src.core.normalizer import normalize_medical_query
from src.core.mmr import compute_mmr, compute_recency_scores
from src.core.utils import set_random_seed, save_run_manifest
from src.ner.ner_service import NERService
from src.encoder.medcpt_encoder import MedCPTEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranker.cross_encoder import CrossEncoderReranker
from src.llm.openai_client import OpenAIClient
from src.llm.stub_llm import StubLLM


class MedicalRAGPipeline:
    """End-to-end Medical RAG Pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        
        # Set random seed for reproducibility
        seed = config.get("pipeline", {}).get("seed", 42)
        set_random_seed(seed)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        # NER
        ner_config = self.config.get("ner", {})
        self.ner = NERService(
            model_name=ner_config.get("model", "en_core_sci_sm"),
            confidence_threshold=ner_config.get("confidence_threshold", 0.7)
        )
        
        # Encoder
        encoder_config = self.config.get("encoder", {})
        self.encoder = MedCPTEncoder(
            model_name=encoder_config.get("model", "ncbi/MedCPT-Query-Encoder"),
            device=encoder_config.get("device", "cpu")
        )
        
        # FAISS Index
        faiss_config = self.config.get("faiss", {})
        self.faiss_index = FAISSIndex(
            index_path=faiss_config.get("save_path"),
            embedding_dim=encoder_config.get("embedding_dim", 768)
        )
        
        # BM25 Retriever
        bm25_config = self.config.get("bm25", {})
        self.bm25_retriever = BM25Retriever(
            host=bm25_config.get("elasticsearch_host", "localhost"),
            port=bm25_config.get("elasticsearch_port", 9200),
            index_name=bm25_config.get("index_name", "medical_docs")
        )
        
        # Hybrid Retriever
        self.hybrid_retriever = HybridRetriever(
            faiss_index=self.faiss_index,
            bm25_retriever=self.bm25_retriever,
            alpha=0.5
        )
        
        # Reranker
        reranker_config = self.config.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=reranker_config.get("model", "pritamdeka/S-PubMedBert-MS-MARCO"),
            batch_size=reranker_config.get("batch_size", 16)
        )
        
        # LLM
        llm_config = self.config.get("llm", {})
        llm_provider = llm_config.get("provider", "openai")
        
        if llm_provider == "stub" or os.getenv("LLM_PROVIDER") == "stub":
            self.llm = StubLLM()
        else:
            self.llm = OpenAIClient(
                model=llm_config.get("model", "gpt-4"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 1024)
            )

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Build indices for dense (FAISS) and sparse (BM25) retrieval"""
        if not documents:
            return
        # Encode abstracts for FAISS
        abstracts = [doc.get("abstract", "") for doc in documents]
        embeddings = self.encoder.encode(abstracts)
        self.faiss_index.add_vectors(embeddings)
        # Index documents into Elasticsearch
        try:
            self.bm25_retriever.index_documents(documents)
        except Exception as e:
            print(f"Warning: BM25 indexing failed: {e}")
    
    def process_query(
        self,
        query_text: str,
        top_k: int = 10,
        use_mmr: bool = True,
        recency_boost: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query_text: User query
            top_k: Number of final documents
            use_mmr: Whether to apply MMR
            recency_boost: Whether to boost recent documents
        
        Returns:
            Dictionary with answer and retrieved documents
        """
        run_manifest_id = str(uuid.uuid4())
        
        # 1. Normalize query
        normalized_query = normalize_medical_query(query_text)
        
        # 2. Extract entities (NER)
        entities = self.ner.extract_entities(normalized_query)
        
        # 3. Encode query
        query_embedding = self.encoder.encode_query(normalized_query)
        
        # 4. Hybrid retrieval
        retrieval_config = self.config.get("retrieval", {})
        retrieved_docs = self.hybrid_retriever.retrieve(
            query=normalized_query,
            query_embedding=query_embedding,
            top_k_dense=retrieval_config.get("top_k_dense", 100),
            top_k_sparse=retrieval_config.get("top_k_sparse", 100),
            top_k_final=retrieval_config.get("top_k_final", 50)
        )
        
        # 5. Rerank
        reranker_config = self.config.get("reranker", {})
        reranked_docs = self.reranker.rerank(
            query=normalized_query,
            documents=retrieved_docs,
            top_k=reranker_config.get("top_k", 20)
        )
        
        # 6. Apply MMR for diversity
        if use_mmr and len(reranked_docs) > 0:
            mmr_config = self.config.get("mmr", {})
            doc_embeddings = self.encoder.encode(
                [doc.get("abstract", "") for doc in reranked_docs]
            )
            
            # Compute recency scores if needed
            recency_scores = None
            if recency_boost:
                pub_dates = [doc.get("pub_date", "2000-01-01") for doc in reranked_docs]
                recency_scores = compute_recency_scores(pub_dates)
            
            selected_indices = compute_mmr(
                query_embedding=query_embedding,
                candidate_embeddings=doc_embeddings,
                lambda_param=mmr_config.get("lambda_param", 0.7),
                top_k=top_k,
                recency_scores=recency_scores,
                recency_weight=mmr_config.get("recency_weight", 0.3) if recency_boost else 0.0
            )
            
            final_docs = [reranked_docs[i] for i in selected_indices]
        else:
            final_docs = reranked_docs[:top_k]
        
        # 7. Generate answer with LLM
        llm_config = self.config.get("llm", {})
        system_prompt = llm_config.get("system_prompt", "You are a medical AI assistant.")
        
        answer = self.llm.generate_with_context(
            query=query_text,
            context_documents=final_docs,
            system_prompt=system_prompt
        )
        
        # 8. Format response
        return {
            "query": query_text,
            "normalized_query": normalized_query,
            "entities": entities,
            "answer": answer,
            "retrieved_documents": final_docs,
            "run_manifest_id": run_manifest_id,
            "metadata": {
                "num_retrieved": len(retrieved_docs),
                "num_reranked": len(reranked_docs),
                "num_final": len(final_docs)
            }
        }
