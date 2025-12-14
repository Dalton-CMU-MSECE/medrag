# Running RAG Pipeline with BioASQ Data

This guide explains how to run the medical RAG pipeline with BioASQ Synergy task data.

## Quick Start

### 1. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare BioASQ data

Place your BioASQ data files in `data/bioasq/`:
- `testset_1.json`, `testset_2.json`, etc.
- `golden_round_1.json`, `golden_round_2.json`, etc.
- `feedback_accompanying_round_1.json`, etc.

### 3. Run the complete pipeline

```bash
# Run on BioASQ round 1 with LLM judge evaluation
python scripts/run_bioasq_pipeline.py \
    --round 1 \
    --email your.email@example.com \
    --output results/round_1 \
    --use-llm-judge
```

### 4. View results

Results will be saved to `results/round_1/`:
- `predictions.json`: Generated answers and retrieved documents
- `metrics.json`: Evaluation metrics

## Pipeline Components

### 1. Data Loading (`BioASQDataLoader`)

Loads BioASQ testset, golden, and feedback files:

```python
from src.core.bioasq_loader import BioASQDataLoader

loader = BioASQDataLoader("data/bioasq")
testset = loader.load_testset(round_num=1)
golden = loader.load_golden(round_num=1)
```

### 2. PubMed Fetching (`PubMedFetcher`)

Fetches abstracts from PubMed using Entrez API:

```python
from src.core.pubmed_fetcher import PubMedFetcher

fetcher = PubMedFetcher(email="your.email@example.com")
pmids = ["12345678", "87654321"]
articles = fetcher.fetch_abstracts(pmids)
```

**Important:** Provide a valid email for NCBI Entrez API compliance.

### 3. RAG Pipeline (`MedicalRAGPipeline`)

Multi-stage pipeline with NER, retrieval, reranking, and MMR:

```python
from src.pipeline.med_rag import MedicalRAGPipeline

pipeline = MedicalRAGPipeline("configs/pipeline_config.yaml")
pipeline.index_documents(documents)

result = pipeline.query("What is the role of HIF-1α in cancer?")
# Returns: answer, retrieved_documents, reranked_documents, final_documents
```

### 4. Evaluation (`RAGEvaluator`)

Evaluates retrieval and answer quality:

```python
from evaluation.evaluation_QA_system.RAG_evaluator import RAGEvaluator

# With LLM judge
evaluator = RAGEvaluator(use_llm_judge=True, llm_judge_model="gpt-4")
metrics = evaluator.evaluate_batch(predictions, ground_truth)

# Metrics include:
# - avg_recall@k, avg_precision@k, avg_mrr
# - avg_rouge1, avg_rouge2, avg_rougeL
# - llm_judge_factuality, llm_judge_completeness, etc.
```

## Configuration

### Pipeline Configuration (`configs/pipeline_config.yaml`)

```yaml
# NER settings
ner:
  model: "en_ner_bc5cdr_md"  # SciSpacy model

# Encoder settings
encoder:
  model: "ncbi/MedCPT-Query-Encoder"
  device: "cuda"

# Retrieval settings
retrieval:
  faiss_top_k: 100
  bm25_top_k: 100
  hybrid_weight: 0.5

# Reranker settings
reranker:
  model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
  top_k: 50

# MMR settings (NEW)
mmr:
  lambda_param: 0.7  # Balance relevance vs diversity
  top_k: 20
  use_recency: true
  recency_weight: 0.3

# LLM settings
llm:
  model: "gpt-4"
  temperature: 0.0
  max_tokens: 500

# Evaluation settings (NEW)
evaluation:
  use_llm_judge: true
  llm_judge_model: "gpt-4"
```

## Advanced Usage

### Testing with a subset of questions

```bash
python scripts/run_bioasq_pipeline.py \
    --round 1 \
    --email your.email@example.com \
    --max-questions 10 \
    --output results/test
```

### Running without LLM judge (faster)

```bash
python scripts/run_bioasq_pipeline.py \
    --round 1 \
    --email your.email@example.com \
    --output results/round_1
```

### Using different configuration

```bash
python scripts/run_bioasq_pipeline.py \
    --round 1 \
    --email your.email@example.com \
    --config configs/pipeline_config_custom.yaml \
    --output results/custom
```

### Batch processing all rounds

```bash
for round in 1 2 3 4; do
    python scripts/run_bioasq_pipeline.py \
        --round $round \
        --email your.email@example.com \
        --output results/round_$round \
        --use-llm-judge
done
```

## Pipeline Stages

### Stage 1: Named Entity Recognition
- Extracts diseases, chemicals, genes from query
- Uses SciSpacy BC5CDR model

### Stage 2: Initial Retrieval
- **FAISS**: Dense retrieval with MedCPT encoder
- **BM25**: Sparse keyword-based retrieval
- **Hybrid**: Combines both with configurable weight

### Stage 3: Cross-Encoder Reranking
- Re-scores documents with S-PubMedBERT
- More accurate relevance scoring

### Stage 4: MMR (Maximal Marginal Relevance) [NEW]
- Balances relevance and diversity
- Optionally considers recency (publication date)
- Reduces redundancy in retrieved documents

### Stage 5: LLM Generation
- Generates answer using top-k documents
- Grounded in retrieved evidence

## Evaluation Metrics

### Retrieval Metrics
- **Recall@k**: Fraction of relevant docs in top-k
- **Precision@k**: Fraction of retrieved docs that are relevant
- **MRR**: Mean Reciprocal Rank

### Answer Quality Metrics
- **ROUGE-1/2/L**: N-gram overlap with ideal answer
- **BLEU**: Precision-based metric

### LLM Judge Metrics [NEW]
- **Factuality (0-1)**: Correctness of information
- **Completeness (0-1)**: Coverage of key points
- **Relevance (0-1)**: Addresses the question
- **Evidence Support (0-1)**: Grounded in retrieved docs
- **Overall Score (0-1)**: Holistic quality

## New Features (Paper Expansion)

### 1. MMR for Diversity
- **Why**: Reduces redundancy, improves coverage
- **How**: Penalizes similarity to already-selected docs
- **Config**: `mmr.lambda_param` (0.7 = balanced)

### 2. LLM-as-a-Judge Evaluation
- **Why**: Captures nuanced answer quality beyond n-grams
- **How**: GPT-4 evaluates on 4 aspects + overall
- **Config**: `evaluation.use_llm_judge = true`

### 3. Recency in MMR
- **Why**: Prioritize recent research in biomedicine
- **How**: Boosts scores based on publication date
- **Config**: `mmr.use_recency = true`

## Troubleshooting

### Issue: NCBI Entrez API rate limiting
**Solution**: The `PubMedFetcher` includes automatic rate limiting (3 requests/sec). For large-scale fetching, consider using NCBI E-utilities with an API key.

### Issue: CUDA out of memory
**Solution**: Reduce batch size in encoder/reranker, or use `device: "cpu"` in config.

### Issue: Missing ideal_answer in evaluation
**Solution**: Some BioASQ questions (yesno, list) may not have ideal_answer. The evaluator handles this gracefully.

### Issue: Low retrieval recall
**Solution**: 
1. Increase `retrieval.faiss_top_k` and `retrieval.bm25_top_k`
2. Adjust `retrieval.hybrid_weight` (0.5 = equal balance)
3. Check if PubMed abstracts were fetched correctly

## Citation

If you use this system, please cite:

```bibtex
@inproceedings{medical_rag_2024,
  title={A Multi-Stage RAG Pipeline for Biomedical Semantic Question Answering},
  author={Your Name},
  booktitle={CEUR Workshop Proceedings},
  volume={4038},
  year={2024}
}
```

## References

- BioASQ Challenge: http://bioasq.org/
- MedCPT: https://arxiv.org/abs/2307.00589
- S-PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
- SciSpacy: https://allenai.github.io/scispacy/
