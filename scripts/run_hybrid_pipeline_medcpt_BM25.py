"""
Run the MedCPT+BM25 hybrid RAG pipeline on BioASQ data

Usage:
  python scripts/run_bioasq_pipeline_hybrid_medcpt.py --round 1 --email user@example.com --config configs/pipeline_config.yaml --data-dir data/bioasq --output results --max-questions 10
"""

import argparse
import json
import os
from typing import Dict, List, Any
import logging

from src.core.bioasq_loader import BioASQDataLoader
from src.core.pubmed_fetcher import PubMedFetcher
from src.pipeline.med_rag_hybrid_medcpt import MedicalRAGPipelineHybridMedCPT
from evaluation.evaluation_QA_system.RAG_evaluator import RAGEvaluator
from src.llm.openai_client import OpenAIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_bioasq_data(round_num: int, data_dir: str) -> Dict[str, Any]:
    loader = BioASQDataLoader(data_dir)
    logger.info(f"Loading BioASQ round {round_num} data...")
    testset = loader.load_testset(round_num)
    golden = loader.load_golden(round_num)
    return {"testset": testset, "golden": golden, "questions": testset["questions"]}


def load_processed_eval(round_num: int, processed_dir: str = "data/processed") -> List[Dict[str, Any]]:
    """
    Load processed evaluation dataset for a specific round.
    """
    eval_path = os.path.join(processed_dir, f"bioasq_round_{round_num}_eval.json")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Processed eval file not found: {eval_path}")
    with open(eval_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Processed eval file must be a list: {eval_path}")
    return data


def fetch_pubmed_docs_from_pmids(pmids: List[str], email: str) -> Dict[str, Dict[str, Any]]:
    fetcher = PubMedFetcher(email=email)
    unique_pmids = list(dict.fromkeys(pmids))
    logger.info(f"Fetching {len(unique_pmids)} PubMed articles...")
    return fetcher.fetch_abstracts(unique_pmids)


def prepare_documents(articles: Dict[str, Any]) -> List[Dict[str, Any]]:
    documents = []
    for pmid, article in articles.items():
        if not article:
            continue
        if isinstance(article, str):
            title, abstract, pub_date, authors = "", article.strip(), None, []
        elif isinstance(article, dict):
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            pub_date = article.get("pub_date")
            authors = article.get("authors", [])
        else:
            continue
        if not abstract:
            continue
        documents.append({
            "doc_id": pmid,
            "text": f"{title}\n\n{abstract}".strip(),
            "title": title,
            "abstract": abstract,
            "pub_date": pub_date,
            "metadata": {"pmid": pmid, "authors": authors, "pub_date": pub_date}
        })
    return documents


def run_pipeline(questions: List[Dict[str, Any]], documents: List[Dict[str, Any]], config_path: str) -> List[Dict[str, Any]]:
    logger.info("Initializing MedCPT+BM25 hybrid RAG pipeline...")
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    pipeline = MedicalRAGPipelineHybridMedCPT(config)
    logger.info(f"Indexing {len(documents)} documents into FAISS and Elasticsearch...")
    pipeline.index_documents(documents)
    predictions = []
    logger.info(f"Processing {len(questions)} questions...")
    for i, q in enumerate(questions):
        if i % 10 == 0:
            logger.info(f"Processing question {i+1}/{len(questions)}")
        query = q["body"]
        qid = q["id"]
        result = pipeline.process_query(query)
        predictions.append({
            "question_id": qid,
            "question_text": query,
            "answer": result["answer"],
            "retrieved_documents": result["retrieved_documents"],
            "reranked_documents": result.get("reranked_documents", []),
            "final_documents": result.get("final_documents", [])
        })
    return predictions


def evaluate_results(predictions: List[Dict[str, Any]], ground_truth_source: Any, use_llm_judge: bool = False, source_type: str = "golden") -> Dict[str, float]:
    evaluator = RAGEvaluator(use_llm_judge=use_llm_judge)
    ground_truth = []
    if source_type == "golden":
        gq = {q["id"]: q for q in ground_truth_source.get("questions", [])}
        for pred in predictions:
            qid = pred["question_id"]
            if qid in gq:
                golden_q = gq[qid]
                ground_truth.append({
                    "question_id": qid,
                    "question_text": pred["question_text"],
                    "type": golden_q.get("type"),
                    "exact_answer": golden_q.get("exact_answer"),
                    "relevant_docs": golden_q.get("documents", []),
                    "ideal_answer": golden_q.get("ideal_answer")
                })
    else:
        pq_map = {q["question_id"]: q for q in ground_truth_source}
        for pred in predictions:
            qid = pred["question_id"]
            if qid in pq_map:
                pq = pq_map[qid]
                ground_truth.append({
                    "question_id": qid,
                    "question_text": pred["question_text"],
                    "type": pq.get("question_type"),
                    "exact_answer": pq.get("exact_answer"),
                    "relevant_docs": pq.get("relevant_docs", []),
                    "ideal_answer": pq.get("ideal_answer")
                })
    logger.info("Evaluating predictions (Hybrid MedCPT+BM25)...")
    return evaluator.evaluate_batch(predictions, ground_truth)


def save_results(predictions: List[Dict[str, Any]], metrics: Dict[str, float], output_dir: str, round_num: int):
    os.makedirs(os.path.join(output_dir, f"round_{round_num}"), exist_ok=True)
    base_dir = os.path.join(output_dir, f"round_{round_num}")
    pred_file = os.path.join(base_dir, "predictions_hybrid_medcpt.json")
    metrics_file = os.path.join(base_dir, "metrics_hybrid_medcpt.json")
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved Hybrid MedCPT predictions to {pred_file}")
    logger.info(f"Saved Hybrid MedCPT metrics to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Run MedCPT+BM25 hybrid RAG pipeline on BioASQ data")
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--email", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/bioasq")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--eval-source", type=str, choices=["golden", "processed"], default="processed")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set")
    _ = OpenAIClient()  # warm-up

    data = load_bioasq_data(args.round, args.data_dir)
    questions = data["questions"]
    if args.max_questions:
        questions = questions[: args.max_questions]
        logger.info(f"Limited to {args.max_questions} questions for testing")

    # Determine PMIDs from chosen eval source
    if args.eval_source == "golden":
        pmids = []
        for q in data["golden"].get("questions", []):
            pmids.extend(q.get("documents", []))
    else:
        processed_eval = load_processed_eval(args.round, args.processed_dir)
        pmids = []
        for q in processed_eval:
            pmids.extend(q.get("relevant_docs", []))
    articles = fetch_pubmed_docs_from_pmids(pmids, args.email)
    documents = prepare_documents(articles)
    logger.info(f"Prepared {len(documents)} documents for indexing")

    predictions = run_pipeline(questions, documents, args.config)
    if args.eval_source == "golden":
        metrics = evaluate_results(predictions, data["golden"], args.use_llm_judge, source_type="golden")
    else:
        metrics = evaluate_results(predictions, processed_eval, args.use_llm_judge, source_type="processed")
    save_results(predictions, metrics, args.output, args.round)


if __name__ == "__main__":
    main()
