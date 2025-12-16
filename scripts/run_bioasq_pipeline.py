"""
End-to-end script to run the RAG pipeline with BioASQ data

Usage:
    python scripts/run_bioasq_pipeline.py --round 1 --email user@example.com --output results/round_1

This script:
1. Loads BioASQ testset and golden data
2. Fetches PubMed abstracts
3. Builds FAISS and BM25 indices
4. Runs the RAG pipeline with MMR
5. Evaluates with traditional metrics + LLM judge
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging

from src.core.bioasq_loader import BioASQDataLoader
from src.core.pubmed_fetcher import PubMedFetcher
from src.pipeline.med_rag import MedicalRAGPipeline
from evaluation.evaluation_QA_system.RAG_evaluator import RAGEvaluator
from src.llm.openai_client import OpenAIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_bioasq_data(round_num: int, data_dir: str = "data/bioasq") -> Dict[str, Any]:
    """
    Load BioASQ testset and golden data for a specific round
    
    Args:
        round_num: BioASQ round number (1-4)
        data_dir: Directory containing BioASQ data files
    
    Returns:
        Dictionary with questions and golden data
    """
    loader = BioASQDataLoader(data_dir)
    
    logger.info(f"Loading BioASQ round {round_num} data...")
    testset = loader.load_testset(round_num)
    golden = loader.load_golden(round_num)
    
    return {
        "testset": testset,
        "golden": golden,
        "questions": testset["questions"]
    }


def fetch_pubmed_docs(golden_data: Dict[str, Any], email: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetch PubMed abstracts for all PMIDs in golden data
    
    Args:
        golden_data: BioASQ golden data
        email: Email for NCBI Entrez API
    
    Returns:
        Dictionary mapping PMID to article data
    """
    fetcher = PubMedFetcher(email=email)
    
    # Collect all unique PMIDs
    all_pmids = set()
    for question in golden_data.get("questions", []):
        all_pmids.update(question.get("documents", []))
    
    logger.info(f"Fetching {len(all_pmids)} PubMed articles...")
    articles = fetcher.fetch_abstracts(list(all_pmids))
    # 'articles' is already a dict mapping PMID -> article dict
    return articles


def prepare_documents(articles: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert PubMed articles to document format for indexing
    
    Args:
        articles: Dictionary of PMID to article data
    
    Returns:
        List of documents with text and metadata
    """
    documents = []
    
    for pmid, article in articles.items():
        if not article:
            continue

        # Support both dict and string article formats
        if isinstance(article, str):
            title = ""
            abstract = article.strip()
            pub_date = None
            authors = []
        elif isinstance(article, dict):
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            pub_date = article.get("pub_date")
            authors = article.get("authors", [])
        else:
            # Unknown format; skip
            continue

        if not abstract:
            continue

        documents.append({
            "doc_id": pmid,
            "text": f"{title}\n\n{abstract}".strip(),
            "title": title,
            "abstract": abstract,
            "pub_date": pub_date,
            "metadata": {
                "pmid": pmid,
                "authors": authors,
                "pub_date": pub_date
            }
        })
    
    return documents

# Ensure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "Environment variable OPENAI_API_KEY is not set. "
        "Please export your key: export OPENAI_API_KEY='sk-xxxx'"
    )

# Optional: initialize OpenAIClient globally
openai_client = OpenAIClient()

def run_pipeline(
    questions: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    config_path: str = "configs/pipeline_config.yaml"
) -> List[Dict[str, Any]]:
    """
    Run the RAG pipeline on BioASQ questions
    
    Args:
        questions: List of BioASQ questions
        documents: List of indexed documents
        config_path: Path to pipeline configuration
    
    Returns:
        List of predictions with answers and retrieved documents
    """
    logger.info("Initializing RAG pipeline...")
    # Load YAML configuration from path
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")

    pipeline = MedicalRAGPipeline(config)
    
    # Build indices
    logger.info(f"Indexing {len(documents)} documents...")
    pipeline.index_documents(documents)
    
    # Run queries
    predictions = []
    logger.info(f"Processing {len(questions)} questions...")
    
    for i, question in enumerate(questions):
        if i % 10 == 0:
            logger.info(f"Processing question {i+1}/{len(questions)}")
        
        query = question["body"]
        question_id = question["id"]
        
        # Run pipeline
        # Use pipeline's process_query to run full RAG flow
        result = pipeline.process_query(query)
        
        predictions.append({
            "question_id": question_id,
            "question_text": query,
            "answer": result["answer"],
            "retrieved_documents": result["retrieved_documents"],
            "reranked_documents": result.get("reranked_documents", []),
            "final_documents": result.get("final_documents", [])
        })
    
    return predictions


def evaluate_results(
    predictions: List[Dict[str, Any]],
    golden_data: Dict[str, Any],
    use_llm_judge: bool = False
) -> Dict[str, float]:
    """
    Evaluate predictions against golden data
    
    Args:
        predictions: List of predictions from pipeline
        golden_data: BioASQ golden data with ground truth
        use_llm_judge: Whether to use LLM-as-a-judge evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = RAGEvaluator(use_llm_judge=use_llm_judge)
    
    # Prepare ground truth
    ground_truth = []
    golden_questions = {q["id"]: q for q in golden_data.get("questions", [])}
    
    for pred in predictions:
        qid = pred["question_id"]
        if qid in golden_questions:
            golden_q = golden_questions[qid]
            ground_truth.append({
                "question_id": qid,
                "question_text": pred["question_text"],
                "relevant_docs": golden_q.get("documents", []),
                "ideal_answer": golden_q.get("ideal_answer")
            })
    
    logger.info("Evaluating predictions...")
    metrics = evaluator.evaluate_batch(predictions, ground_truth)
    
    return metrics


def save_results(
    predictions: List[Dict[str, Any]],
    metrics: Dict[str, float],
    output_dir: str
):
    """
    Save predictions and evaluation metrics
    
    Args:
        predictions: List of predictions
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    pred_file = os.path.join(output_dir, "predictions.json")
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Saved predictions to {pred_file}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in sorted(metrics.items()):
        print(f"{key:40s}: {value:.4f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline on BioASQ data")
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3, 4],
                       help="BioASQ round number")
    parser.add_argument("--email", type=str, required=True,
                       help="Email for NCBI Entrez API")
    parser.add_argument("--data-dir", type=str, default="data/bioasq",
                       help="Directory containing BioASQ data")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml",
                       help="Path to pipeline configuration")
    parser.add_argument("--use-llm-judge", action="store_true",
                       help="Use LLM-as-a-judge for evaluation")
    parser.add_argument("--max-questions", type=int, default=None,
                       help="Maximum number of questions to process (for testing)")
    
    args = parser.parse_args()
    
    # Load BioASQ data
    data = load_bioasq_data(args.round, args.data_dir)
    
    # Limit questions if specified
    questions = data["questions"]
    if args.max_questions:
        questions = questions[:args.max_questions]
        logger.info(f"Limited to {args.max_questions} questions for testing")
    
    # Fetch PubMed documents
    articles = fetch_pubmed_docs(data["golden"], args.email)
    
    # Prepare documents
    documents = prepare_documents(articles)
    logger.info(f"Prepared {len(documents)} documents for indexing")
    
    # Run pipeline
    predictions = run_pipeline(questions, documents, args.config)
    
    # Evaluate
    metrics = evaluate_results(predictions, data["golden"], args.use_llm_judge)
    
    # Save results
    output_dir = os.path.join(args.output, f"round_{args.round}")
    save_results(predictions, metrics, output_dir)


if __name__ == "__main__":
    main()
