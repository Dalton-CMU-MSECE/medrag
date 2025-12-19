"""
Run the FAISS-only RAG pipeline on BioASQ data

Usage:
  python scripts/run_bioasq_pipeline_faiss.py --round 1 --email user@example.com --config configs/pipeline_config.yaml --data-dir data --output results --max-questions 10
"""

import argparse
import json
import os
from typing import Dict, List, Any
import logging

from src.core.bioasq_loader import BioASQDataLoader
from src.core.pubmed_fetcher import PubMedFetcher
from src.pipeline.med_rag_faiss import MedicalRAGPipelineFAISS
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


def fetch_pubmed_docs(golden_data: Dict[str, Any], email: str) -> Dict[str, Dict[str, Any]]:
    fetcher = PubMedFetcher(email=email)
    all_pmids = set()
    for q in golden_data.get("questions", []):
        all_pmids.update(q.get("documents", []))
    logger.info(f"Fetching {len(all_pmids)} PubMed articles...")
    return fetcher.fetch_abstracts(list(all_pmids))


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
    logger.info("Initializing FAISS-only RAG pipeline...")
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    pipeline = MedicalRAGPipelineFAISS(config)
    logger.info(f"Indexing {len(documents)} documents into FAISS...")
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
        })
    return predictions


def evaluate_results(predictions: List[Dict[str, Any]], golden_data: Dict[str, Any], use_llm_judge: bool = False) -> Dict[str, float]:
    evaluator = RAGEvaluator(use_llm_judge=use_llm_judge)
    ground_truth = []
    gq = {q["id"]: q for q in golden_data.get("questions", [])}
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
    logger.info("Evaluating predictions (FAISS)...")
    return evaluator.evaluate_batch(predictions, ground_truth)


def save_results(predictions: List[Dict[str, Any]], metrics: Dict[str, float], output_dir: str, round_num: int):
    os.makedirs(os.path.join(output_dir, f"round_{round_num}"), exist_ok=True)
    base_dir = os.path.join(output_dir, f"round_{round_num}")
    pred_file = os.path.join(base_dir, "predictions_faiss.json")
    metrics_file = os.path.join(base_dir, "metrics_faiss.json")
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved FAISS predictions to {pred_file}")
    logger.info(f"Saved FAISS metrics to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Run FAISS-only RAG pipeline on BioASQ data")
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--email", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/bioasq")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set")
    _ = OpenAIClient()  # warm-up

    data = load_bioasq_data(args.round, args.data_dir)
    questions = data["questions"]
    if args.max_questions:
        questions = questions[: args.max_questions]
        logger.info(f"Limited to {args.max_questions} questions for testing")

    articles = fetch_pubmed_docs(data["golden"], args.email)
    documents = prepare_documents(articles)
    logger.info(f"Prepared {len(documents)} documents for indexing")

    predictions = run_pipeline(questions, documents, args.config)
    metrics = evaluate_results(predictions, data["golden"], args.use_llm_judge)
    save_results(predictions, metrics, args.output, args.round)


if __name__ == "__main__":
    main()
