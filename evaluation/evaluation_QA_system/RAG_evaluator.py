"""
RAG System Evaluator for BioASQ data
Includes traditional retrieval metrics + answer-type specific metrics + optional LLM-as-a-judge
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
from src.llm.llm_judge import LLMJudge
import re


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

        # F1@K (harmonic mean of precision and recall)
        for k in k_values:
            p = metrics.get(f"precision@{k}", 0.0)
            r = metrics.get(f"recall@{k}", 0.0)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            metrics[f"f1@{k}"] = f1
        
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

    # ---------------- Answer-type helpers -----------------
    @staticmethod
    def _extract_response(answer_str: str) -> str:
        """Extract 'response' from JSON answer if present, else return raw string."""
        try:
            s = (answer_str or "").strip()
            if s.startswith("{"):
                obj = json.loads(s)
                resp = obj.get("response")
                if isinstance(resp, str):
                    return resp.strip()
        except Exception:
            pass
        return (answer_str or "").strip()

    @staticmethod
    def _flatten(seq: Any) -> List[Any]:
        if seq is None:
            return []
        if isinstance(seq, (str, bytes)):
            return [seq]
        out: List[Any] = []
        stack = [seq]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple, set)):
                stack.extend(item)
            else:
                out.append(item)
        return out

    @staticmethod
    def _norm_text(t: Any) -> str:
        if isinstance(t, (list, tuple, set)):
            parts = [RAGEvaluator._norm_text(x) for x in RAGEvaluator._flatten(t)]
            return " ".join(p for p in parts if p).strip()
        t = ("" if t is None else str(t)).lower().strip()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[\.,;:\-\(\)\[\]{}]", "", t)
        return t

    @staticmethod
    def _split_list(s: str) -> List[str]:
        s = (s or "")
        parts = re.split(r"[,;\n]", s)
        items = [RAGEvaluator._norm_text(p) for p in parts if RAGEvaluator._norm_text(p)]
        # dedupe while preserving order
        seen = set()
        result = []
        for it in items:
            if it not in seen:
                seen.add(it)
                result.append(it)
        return result

    # Yes/No metrics accumulation
    @staticmethod
    def _accumulate_yesno_counts(pred: str, gold: str, counts: Dict[str, int]):
        p = RAGEvaluator._norm_text(pred)
        g = RAGEvaluator._norm_text(gold)
        if p not in ("yes", "no"):
            # simple heuristic: map common forms
            if "no" in p:
                p = "no"
            elif "yes" in p:
                p = "yes"
        # Update confusion-like counts for each class
        # For class 'yes'
        if g == "yes":
            if p == "yes":
                counts["tp_yes"] += 1
            else:
                counts["fn_yes"] += 1
        else:
            if p == "yes":
                counts["fp_yes"] += 1
        # For class 'no'
        if g == "no":
            if p == "no":
                counts["tp_no"] += 1
            else:
                counts["fn_no"] += 1
        else:
            if p == "no":
                counts["fp_no"] += 1
        # Accuracy per sample
        if p == g:
            counts["correct"] += 1
        counts["total"] += 1

    @staticmethod
    def _finalize_yesno_metrics(counts: Dict[str, int]) -> Dict[str, float]:
        def f1(tp, fp, fn):
            denom = (2*tp + fp + fn)
            return (2*tp / denom) if denom > 0 else 0.0
        acc = (counts["correct"] / counts["total"]) if counts["total"] > 0 else 0.0
        f1_yes = f1(counts["tp_yes"], counts["fp_yes"], counts["fn_yes"])
        f1_no = f1(counts["tp_no"], counts["fp_no"], counts["fn_no"])
        macro = (f1_yes + f1_no) / 2.0
        return {
            "yesno_accuracy": acc,
            "yesno_f1_yes": f1_yes,
            "yesno_f1_no": f1_no,
            "yesno_macro_f1": macro,
        }

    # Factoid metrics per sample
    @staticmethod
    def _factoid_metrics(pred: str, gold_list: List[Any]) -> tuple[float, float, float]:
        p = RAGEvaluator._norm_text(pred)
        # Flatten nested gold answers
        flat_gold = RAGEvaluator._flatten(gold_list)
        gold_norms = [RAGEvaluator._norm_text(g) for g in flat_gold]
        strict = 1.0 if any(p == g for g in gold_norms) else 0.0
        # Lenient: substring either way
        lenient = 1.0 if any((p and g and (p in g or g in p)) for g in gold_norms) else 0.0
        # MRR: single prediction — 1 if strict match else 0
        mrr = 1.0 if strict == 1.0 else 0.0
        return strict, lenient, mrr

    # Exact list metrics per sample
    @staticmethod
    def _list_metrics(pred: Any, gold_list: Any) -> tuple[float, float, float]:
        # Normalize prediction string into items
        pred_items = set(RAGEvaluator._split_list(pred if isinstance(pred, str) else RAGEvaluator._norm_text(pred)))
        # Flatten and normalize gold list into unique items
        flat_gold = RAGEvaluator._flatten(gold_list)
        gold_items = set(RAGEvaluator._norm_text(g) for g in flat_gold if RAGEvaluator._norm_text(g))
        tp = len(pred_items & gold_items)
        precision = (tp / len(pred_items)) if pred_items else 0.0
        recall = (tp / len(gold_items)) if gold_items else 0.0
        f1 = (2*precision*recall / (precision+recall)) if (precision+recall) > 0 else 0.0
        return precision, recall, f1

    # Ideal answer metrics per sample
    def _ideal_metrics(self, pred: str, ref: str) -> Dict[str, float]:
        pred_text = (pred or "").strip()
        ref_text = (ref or "").strip()
        # Rouge-1 recall & precision only
        r = self.compute_rouge_scores(pred_text, ref_text)
        # Readability: simple proxy (shorter sentences => higher score)
        sentences = re.split(r"[\.!?]", pred_text)
        words = [w for w in re.split(r"\s+", pred_text) if w]
        avg_len = (len(words) / max(1, len([s for s in sentences if s.strip()])))
        readability = max(0.0, min(1.0, 1.0 - (avg_len - 15) / 35))  # maps ~15-50 words/sentence into 1→0
        # Repetition: ratio of repeated tokens
        norm_tokens = [self._norm_text(w) for w in words]
        total = len(norm_tokens)
        unique = len(set(norm_tokens))
        repetition = (1.0 - (unique / total)) if total > 0 else 0.0
        return {
            "ideal_readability": readability,
            "ideal_rouge1_recall": r.get("rouge1", 0.0),
            "ideal_rouge1_precision": r.get("rouge1", 0.0),  # rouge_scorer returns fmeasure only in current helper
            "ideal_repetition": repetition,
        }
    
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
        # Accumulators for answer-type metrics
        # yes/no
        yn_counts = {"tp_yes":0,"fp_yes":0,"fn_yes":0,"tp_no":0,"fp_no":0,"fn_no":0,"correct":0,"total":0}
        # factoid
        factoid_strict_sum = 0.0
        factoid_lenient_sum = 0.0
        factoid_mrr_sum = 0.0
        factoid_n = 0
        # list
        list_precisions = []
        list_recalls = []
        list_f1s = []
        # ideal
        ideal_metrics_list = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Retrieval metrics
            retrieved_docs = [doc.get("doc_id") or doc.get("pmid") 
                            for doc in pred.get("retrieved_documents", [])]
            relevant_docs = gt.get("relevant_docs", [])
            
            if relevant_docs:
                ret_metrics = self.evaluate_retrieval(retrieved_docs, relevant_docs)
                all_retrieval_metrics.append(ret_metrics)
            
            # Answer-type specific metrics
            if "answer" in pred:
                answer_text = self._extract_response(pred["answer"])  # from JSON if present
                qtype = gt.get("type")
                exact_answer = gt.get("exact_answer")
                ideal_answer = gt.get("ideal_answer")

                if qtype == "yesno" and isinstance(exact_answer, str):
                    self._accumulate_yesno_counts(answer_text, exact_answer, yn_counts)

                elif qtype == "factoid":
                    gold_list = exact_answer if isinstance(exact_answer, list) else ([exact_answer] if exact_answer else [])
                    s, l, m = self._factoid_metrics(answer_text, gold_list)
                    factoid_strict_sum += s
                    factoid_lenient_sum += l
                    factoid_mrr_sum += m
                    factoid_n += 1

                elif qtype == "list":
                    gold_list = exact_answer if isinstance(exact_answer, list) else []
                    p, r, f1 = self._list_metrics(answer_text, gold_list)
                    list_precisions.append(p)
                    list_recalls.append(r)
                    list_f1s.append(f1)

                # Ideal answers use ROUGE-based metrics and readability/repetition
                if ideal_answer:
                    if isinstance(ideal_answer, list):
                        ideal_answer_text = " ".join(ideal_answer)
                    else:
                        ideal_answer_text = str(ideal_answer)
                    im = self._ideal_metrics(answer_text, ideal_answer_text)
                    ideal_metrics_list.append(im)

                # Optional LLM judge
                if self.use_llm_judge and self.llm_judge:
                    snippets = [doc.get("snippet") or doc.get("abstract", "")[:500]
                                for doc in pred.get("retrieved_documents", [])[:5]]
                    llm_eval = self.llm_judge.evaluate_answer(
                        question=gt.get("question_text", ""),
                        generated_answer=answer_text,
                        reference_answer=ideal_answer_text if ideal_answer else None,
                        retrieved_snippets=snippets
                    )
                    all_llm_judge_scores.append(llm_eval)
        
        # Aggregate metrics
        aggregated = {}
        # Yes/No
        yn = self._finalize_yesno_metrics(yn_counts)
        aggregated.update(yn)

        # Factoid
        if factoid_n > 0:
            aggregated["factoid_strict_accuracy"] = factoid_strict_sum / factoid_n
            aggregated["factoid_lenient_accuracy"] = factoid_lenient_sum / factoid_n
            aggregated["factoid_mrr"] = factoid_mrr_sum / factoid_n

        # List
        if list_precisions:
            aggregated["list_mean_precision"] = float(np.mean(list_precisions))
        if list_recalls:
            aggregated["list_mean_recall"] = float(np.mean(list_recalls))
        if list_f1s:
            aggregated["list_mean_f1"] = float(np.mean(list_f1s))

        # Ideal (ROUGE recall/precision + readability + repetition)
        if ideal_metrics_list:
            keys = ideal_metrics_list[0].keys()
            for k in keys:
                aggregated[k] = float(np.mean([m[k] for m in ideal_metrics_list]))
        
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
