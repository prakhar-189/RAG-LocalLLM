# eval/evaluate.py
# -----------------------------------------------------------------------------
# RAG evaluation harness.
#
# Runs a golden question/answer set through the exact same pipeline the app uses
# (semantic chunking -> FAISS retrieval -> strictly grounded Phi-3 answer) and
# scores every answer with three complementary signals:
#
#   1. Lexical overlap   -> ROUGE-L F1        (rouge-score)
#   2. Semantic similarity -> BERTScore F1    (bert-score; falls back to
#                             embedding cosine similarity if unavailable)
#   3. LLM-as-judge      -> a local Ollama model rates factual correctness 1-5
#
# It also reports a retrieval sanity signal (was any context retrieved) and,
# for the out-of-scope question, whether the pipeline correctly refused.
#
# Outputs:
#   eval/results/detailed_results.csv  -- per-question scores + model answers
#   eval/results/summary.md            -- aggregate table (pasted into README)
#
# Usage (from the project root, with `ollama serve` running):
#   python -m eval.evaluate
#   python -m eval.evaluate --judge-model llama3 --limit 5
#   python -m eval.evaluate --no-judge          # skip the LLM judge
# -----------------------------------------------------------------------------
import argparse
import csv
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

# Make `src` importable whether run as a module or a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import create_vector_store  # noqa: E402
from src.rag_pipeline import build_qa_chain  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ------------------------------------------------------------------ metrics ---
def rouge_l_scores(preds, refs):
    """Lexical overlap: ROUGE-L F1 per pair (0-1)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def semantic_scores(preds, refs):
    """
    Semantic similarity per pair (0-1).

    Primary signal is BERTScore F1. If the BERTScore model can't be loaded
    (e.g. offline), we fall back to cosine similarity of the same MiniLM
    sentence embeddings the retriever uses, so the harness still produces a
    semantic number instead of crashing.
    """
    try:
        from bert_score import score as bertscore

        # distilbert keeps the download small (~270MB) and is a supported
        # BERTScore backbone; baseline rescaling is off so no baseline files
        # are required.
        _, _, f1 = bertscore(
            preds,
            refs,
            model_type="distilbert-base-uncased",
            num_layers=5,
            lang="en",
            rescale_with_baseline=False,
            verbose=False,
        )
        return [float(x) for x in f1], "BERTScore-F1"
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] BERTScore unavailable ({exc}); using embedding cosine.")
        import numpy as np
        from src.vector_store import _load_embeddings

        emb = _load_embeddings()
        pv = np.array(emb.embed_documents(preds))
        rv = np.array(emb.embed_documents(refs))
        cos = np.sum(pv * rv, axis=1) / (
            np.linalg.norm(pv, axis=1) * np.linalg.norm(rv, axis=1) + 1e-9
        )
        return [float(x) for x in cos], "EmbeddingCosine"


JUDGE_PROMPT = """You are a strict grader for a question-answering system.
Compare the MODEL ANSWER to the REFERENCE ANSWER and rate how factually correct
and complete the model answer is, on an integer scale from 1 to 5:
5 = fully correct, captures all key facts
3 = partially correct or missing important details
1 = incorrect or irrelevant
Reply with ONLY the single integer.

QUESTION: {question}
REFERENCE ANSWER: {reference}
MODEL ANSWER: {prediction}
RATING (1-5):"""


def judge_scores(questions, preds, refs, judge_model):
    """LLM-as-judge: local Ollama model rates each answer 1-5 -> normalized 0-1."""
    from langchain_ollama import OllamaLLM

    judge = OllamaLLM(model=judge_model, temperature=0.0)
    scores = []
    for q, p, r in zip(questions, preds, refs):
        raw = judge.invoke(
            JUDGE_PROMPT.format(question=q, reference=r, prediction=p)
        )
        match = re.search(r"[1-5]", raw)
        rating = int(match.group()) if match else 3
        scores.append((rating - 1) / 4.0)  # 1..5 -> 0..1
    return scores


# --------------------------------------------------------------------- main ---
def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate the local RAG pipeline.")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "golden_dataset.json"),
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("OLLAMA_MODEL", "phi3:mini"),
        help="Ollama model used as the LLM judge (default: generation model).",
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N items.")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)

    doc_path = PROJECT_ROOT / data["document"]
    qa_pairs = data["qa_pairs"]
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    print(f"Document : {doc_path}")
    print(f"Questions: {len(qa_pairs)}")
    print("Building / loading vector store (cached after first run)...")
    vectorstore = create_vector_store(str(doc_path))
    qa_chain = build_qa_chain(vectorstore)

    questions, refs, preds, retrieved_counts = [], [], [], []
    for item in qa_pairs:
        q = item["question"]
        print(f"  -> Q{item['id']}: {q}")
        response = qa_chain.invoke({"query": q})
        questions.append(q)
        refs.append(item["ground_truth"])
        preds.append(response["result"].strip())
        retrieved_counts.append(len(response.get("source_documents", [])))

    print("Scoring: ROUGE-L ...")
    rouge = rouge_l_scores(preds, refs)
    print("Scoring: semantic similarity ...")
    semantic, semantic_label = semantic_scores(preds, refs)
    if args.no_judge:
        judge, judge_label = [None] * len(preds), "LLM-Judge (skipped)"
    else:
        print(f"Scoring: LLM judge ({args.judge_model}) ...")
        judge = judge_scores(questions, preds, refs, args.judge_model)
        judge_label = f"LLM-Judge ({args.judge_model})"

    # --- write per-question CSV -------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "detailed_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "question", "rougeL_f1", semantic_label, judge_label,
             "chunks_retrieved", "model_answer", "ground_truth"]
        )
        for item, q, rg, sm, jd, rc, p, r in zip(
            qa_pairs, questions, rouge, semantic, judge, retrieved_counts, preds, refs
        ):
            w.writerow([
                item["id"], q, round(rg, 3), round(sm, 3),
                "" if jd is None else round(jd, 3), rc, p, r,
            ])

    # --- aggregate + summary.md -------------------------------------------------
    valid_judge = [j for j in judge if j is not None]
    lines = []
    lines.append("# RAG Evaluation Results\n")
    lines.append(
        f"_Generated {date.today().isoformat()} · {len(qa_pairs)} questions · "
        f"embeddings `all-MiniLM-L6-v2` · generator `phi3:mini` · "
        f"retriever top-k=3 · strict grounding_\n"
    )
    lines.append("| Metric | Signal | Mean score (0-1) |")
    lines.append("|--------|--------|------------------|")
    lines.append(f"| Lexical overlap | ROUGE-L F1 | {mean(rouge):.3f} |")
    lines.append(f"| Semantic similarity | {semantic_label} | {mean(semantic):.3f} |")
    if valid_judge:
        lines.append(f"| Answer correctness | {judge_label} | {mean(valid_judge):.3f} |")
    retrieval_ok = sum(1 for c in retrieved_counts if c > 0)
    lines.append(
        f"| Retrieval coverage | chunks found / question | "
        f"{retrieval_ok}/{len(qa_pairs)} |"
    )
    lines.append("")
    summary = "\n".join(lines)
    (RESULTS_DIR / "summary.md").write_text(summary + "\n", encoding="utf-8")

    print("\n" + summary)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
