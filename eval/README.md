# RAG Evaluation Harness

This folder measures the *quality* of the RAG pipeline, not just that it runs.
It scores the pipeline's answers against a golden question/answer set with three
complementary signals so no single metric can hide a weakness:

| Signal | Metric | What it catches |
|--------|--------|-----------------|
| Lexical overlap | ROUGE-L F1 | Missing/incorrect wording vs. the reference |
| Semantic similarity | BERTScore F1 (embedding-cosine fallback) | Right meaning, different words |
| Answer correctness | LLM-as-judge (local Ollama) | Factual correctness a keyword metric would miss |

The set also includes an **out-of-scope question** ("What is the capital of
France?") to verify the strictly grounded pipeline correctly *refuses* instead
of hallucinating.

## Files

- `golden_dataset.json` — questions + ground-truth answers grounded in the test PDF.
- `evaluate.py` — runs the pipeline and scores every answer.
- `results/` — generated: `detailed_results.csv` (per question) and `summary.md`
  (the aggregate table shown in the project README).

## Running

From the project root, with `ollama serve` running and `phi3:mini` pulled:

```bash
python -m eval.evaluate                       # full run
python -m eval.evaluate --judge-model llama3  # use a stronger judge model
python -m eval.evaluate --no-judge            # skip the LLM judge (faster)
python -m eval.evaluate --limit 5             # first 5 questions only
```

The first run builds the FAISS index for the document (slow — it embeds every
sentence). Thanks to the on-disk vector cache, subsequent runs load it instantly.
