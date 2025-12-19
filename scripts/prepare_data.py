#!/usr/bin/env python3
"""
Prepare BioASQ data for the RAG pipeline
- Load BioASQ golden data
- Fetch PubMed abstracts
- Create documents corpus
- Save in format compatible with pipeline
"""

import argparse
import json
from pathlib import Path

from src.core.bioasq_loader import BioASQDataLoader
from src.core.pubmed_fetcher import PubMedFetcher


def main():
    parser = argparse.ArgumentParser(description="Prepare BioASQ data for RAG pipeline")
    parser.add_argument("--round", type=int, required=True, help="BioASQ round number (1-4)")
    parser.add_argument("--data-dir", default="data", help="Directory containing BioASQ files")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--email", required=True, help="Email for PubMed API")
    parser.add_argument("--api-key", help="NCBI API key (optional)")
    parser.add_argument("--max-docs", type=int, help="Maximum documents to fetch (for testing)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load BioASQ data
    print(f"Loading BioASQ round {args.round} data...")
    loader = BioASQDataLoader(args.data_dir)
    golden_data = loader.load_golden(args.round)
    
    # Collect all unique PubMed IDs
    print("Collecting PubMed IDs...")
    all_pmids = set()
    for question in golden_data["questions"]:
        pmids = loader.extract_document_ids(question)
        all_pmids.update(pmids)
    
    print(f"Found {len(all_pmids)} unique PubMed documents")
    
    # Limit for testing if specified
    if args.max_docs:
        all_pmids = list(all_pmids)[:args.max_docs]
        print(f"Limited to {len(all_pmids)} documents for testing")
    
    # Fetch PubMed abstracts
    print("Fetching PubMed abstracts...")
    fetcher = PubMedFetcher(email=args.email, api_key=args.api_key)
    documents = fetcher.fetch_abstracts(list(all_pmids))
    
    print(f"Successfully fetched {len(documents)} documents")
    
    # Save documents in JSONL format
    docs_output = output_dir / f"bioasq_round_{args.round}_docs.jsonl"
    print(f"Saving documents to {docs_output}...")
    
    with open(docs_output, 'w', encoding='utf-8') as f:
        for doc in documents.values():
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Create evaluation ground truth file
    print("Creating evaluation ground truth...")
    eval_output = output_dir / f"bioasq_round_{args.round}_eval.json"
    
    eval_data = []
    for question in golden_data["questions"]:
        if question.get("answerReady", False):
            eval_entry = loader.format_for_evaluation(question)
            eval_data.append(eval_entry)
    
    with open(eval_output, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created evaluation file with {len(eval_data)} answer-ready questions")
    
    # Create statistics file
    stats = {
        "round": args.round,
        "total_questions": len(golden_data["questions"]),
        "answer_ready_questions": len(eval_data),
        "total_documents": len(documents),
        "unique_pmids": len(all_pmids),
        "question_types": {}
    }
    
    # Count by question type
    for q in golden_data["questions"]:
        qtype = q.get("type", "unknown")
        stats["question_types"][qtype] = stats["question_types"].get(qtype, 0) + 1
    
    stats_output = output_dir / f"bioasq_round_{args.round}_stats.json"
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("\n=== Statistics ===")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Answer-ready questions: {stats['answer_ready_questions']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Question types: {stats['question_types']}")
    print(f"\nData preparation complete!")
    print(f"Documents: {docs_output}")
    print(f"Evaluation: {eval_output}")
    print(f"Statistics: {stats_output}")


if __name__ == "__main__":
    main()
