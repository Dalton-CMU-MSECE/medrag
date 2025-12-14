"""
RAG System Evaluator for BioASQ data
Includes traditional metrics + LLM-as-a-judge evaluation
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
from src.llm.llm_judge import LLMJudge


class RAGEvaluator:
    """Evaluator for RAG system performance on BioASQ data"""
    
    def __init__(self, use_llm_judge: bool = False, llm_judge_model: str = "gpt-4"):
        """
        Initialize evaluator
        
        Args:
            use_llm_judge: Whether to use LLM-as-a-judge for answer quality
            llm_judge_model: Model to use for LLM judge
        """
        self.use_llm_judge = use_llm_judge
        self.llm_judge = None
        
        if use_llm_judge:
            self.llm_judge = LLMJudge(model=llm_judge_model)
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval metrics
        
        Args:
            retrieved_docs: List of retrieved document IDs (in rank order)
            relevant_docs: List of relevant document IDs
            k_values: K values for recall@k
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Recall@K
        for k in k_values:
            retrieved_at_k = set(retrieved_docs[:k])
            relevant_set = set(relevant_docs)
            recall = len(retrieved_at_k & relevant_set) / len(relevant_set) if relevant_set else 0
            metrics[f"recall@{k}"] = recall
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                mrr = 1.0 / i
                break
        metrics["mrr"] = mrr
        
        # Precision@K
        for k in k_values:
            retrieved_at_k = set(retrieved_docs[:k])
            relevant_set = set(relevant_docs)
            precision = len(retrieved_at_k & relevant_set) / k if k > 0 else 0
            metrics[f"precision@{k}"] = precision
        
        return metrics
    
    def compute_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores for answer quality
        
        Args:
            prediction: Generated answer
            reference: Reference answer
        
        Returns:
            Dictionary of ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, prediction)
            
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except ImportError:
            print("Warning: rouge_score not installed")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def evaluate_bioasq_retrieval(
        self,
        retrieved_docs: List[str],
        golden_docs: List[str],
        k_values: List[int] = [5, 10, 20, 50]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval for BioASQ with standard metrics
        
        Args:
            retrieved_docs: List of retrieved PubMed IDs (in rank order)
            golden_docs: List of golden/relevant PubMed IDs
            k_values: K values for recall@k and precision@k
        
        Returns:
            Dictionary of retrieval metrics
        """
        return self.evaluate_retrieval(retrieved_docs, golden_docs, k_values)
    
    def evaluate_batch(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions for BioASQ
        
        Args:
            predictions: List of prediction dictionaries with:
                - question_id, answer, retrieved_documents
            ground_truth: List of ground truth dictionaries with:
                - question_id, relevant_docs, ideal_answer
        
        Returns:
            Aggregated evaluation metrics including LLM judge scores
        """
        all_retrieval_metrics = []
        all_rouge_scores = []
        all_llm_judge_scores = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Retrieval metrics
            retrieved_docs = [doc.get("doc_id") or doc.get("pmid") 
                            for doc in pred.get("retrieved_documents", [])]
            relevant_docs = gt.get("relevant_docs", [])
            
            if relevant_docs:
                ret_metrics = self.evaluate_retrieval(retrieved_docs, relevant_docs)
                all_retrieval_metrics.append(ret_metrics)
            
            # Answer quality metrics
            if "answer" in pred and gt.get("ideal_answer"):
                # Handle ideal_answer as list or string
                ideal_answer = gt["ideal_answer"]
                if isinstance(ideal_answer, list):
                    ideal_answer = " ".join(ideal_answer)
                
                rouge = self.compute_rouge_scores(pred["answer"], ideal_answer)
                all_rouge_scores.append(rouge)
                
                # LLM Judge evaluation
                if self.use_llm_judge and self.llm_judge:
                    snippets = [doc.get("snippet") or doc.get("abstract", "")[:500]
                              for doc in pred.get("retrieved_documents", [])[:5]]
                    
                    llm_eval = self.llm_judge.evaluate_answer(
                        question=gt.get("question_text", ""),
                        generated_answer=pred["answer"],
                        reference_answer=ideal_answer,
                        retrieved_snippets=snippets
                    )
                    all_llm_judge_scores.append(llm_eval)
        
        # Aggregate metrics
        aggregated = {}
        
        if all_retrieval_metrics:
            for key in all_retrieval_metrics[0].keys():
                values = [m[key] for m in all_retrieval_metrics]
                aggregated[f"avg_{key}"] = np.mean(values)
        
        if all_rouge_scores:
            for key in all_rouge_scores[0].keys():
                values = [s[key] for s in all_rouge_scores]
                aggregated[f"avg_{key}"] = np.mean(values)
        
        # Aggregate LLM judge scores
        if all_llm_judge_scores:
            # Aggregate by aspect
            aspects = ["factuality", "completeness", "relevance", "evidence_support", "overall_score"]
            for aspect in aspects:
                if aspect in all_llm_judge_scores[0]:
                    values = [s[aspect].get("score", s[aspect]) if isinstance(s[aspect], dict) 
                            else s[aspect] for s in all_llm_judge_scores]
                    aggregated[f"llm_judge_{aspect}"] = np.mean(values)
        
        return aggregated
