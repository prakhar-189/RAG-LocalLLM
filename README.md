# GenAI-LangChain-LocalLLM

> 🧠 A robust, local Large Language Model (LLM) chatbot built using LangChain, Streamlit, and the Phi-3 model.

This project enables users to interact with a lightweight generative AI model directly on their local machine, ensuring fast inference, complete data privacy, and zero reliance on external API keys or internet connectivity for query processing.

---

## ✨ Features

- **100% Local Execution** — Run generative AI entirely on your hardware. Keep your conversational data private and secure.
- **Phi-3 Integration** — Leverages the highly efficient and capable Phi-3 model via Ollama, optimized for local environments without sacrificing conversational quality.
- **LangChain Orchestration** — Utilizes the LangChain framework to manage prompts, handle conversational memory, and structure the RAG (Retrieval-Augmented Generation) pipeline efficiently.
- **Interactive UI** — Features a clean, responsive chatbot interface built with Streamlit, making it easy to interact with the model right out of the box.
- **Persistent Vector Cache** — The FAISS index is fingerprinted by document *content* and cached to disk, so re-uploading the same PDF skips the expensive re-embedding step and loads instantly instead of re-processing every sentence.
- **Built-in RAG Evaluation** — A reproducible harness scores answer quality on a golden Q&A set using three complementary signals (lexical overlap, semantic similarity, and an LLM-as-judge), so quality is *measured*, not assumed.

---

## 📂 Repository Structure

```
GenAI-LangChain-LocalLLM/
├── Outputs/                 # Directory containing visual proof of model outputs and citations
│   ├── Output 1/
│   │   ├── Citation 1.png
│   │   ├── Citation 2.png
│   │   ├── LLM Generated Answer.png
│   │   └── Streamlit Home Page & Question.png
│   └── Output 2/
│       ├── Citation 1.png
│       ├── Citation 2.png
│       ├── LLM Generated Answer.png
│       └── Streamlit Home Page & Question.png
├── Test Datasets/           # Sample datasets for evaluating and testing model accuracy
│   └── Data Analytics using SQL.pdf
├── eval/                    # Reproducible RAG evaluation harness
│   ├── golden_dataset.json  # Grounded Q&A set with ground-truth answers
│   ├── evaluate.py          # Three-signal scorer (ROUGE-L + BERTScore + LLM judge)
│   ├── results/             # Generated scores (per-question CSV + summary table)
│   └── README.md            # Evaluation methodology
├── src/                     # Core modules, LangChain utilities, and backend logic
│   ├── llm_model.py         # Handles the initialization and configuration of the local LLM
│   ├── rag_pipeline.py      # Orchestrates the prompt management and generation pipeline
│   └── vector_store.py      # Embeddings, FAISS vector store, and on-disk index caching
├── .gitignore               # Ignored files and directories
├── LICENSE                  # MIT License
├── app.py                   # Main Streamlit application entry point
└── requirements.txt         # List of all Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ installed on your machine.
- A machine capable of running local LLMs (sufficient RAM/VRAM depending on the quantization of the Phi-3 model used).

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/PrakharSri18-data/GenAI-LangChain-LocalLLM.git
cd GenAI-LangChain-LocalLLM
```

**2. Create a virtual environment (Recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install the dependencies:**

```bash
pip install -r requirements.txt
```

**4. Model Setup:**

Ensure you have the local Phi-3 model initialized (e.g., via Ollama) and available to the scripts in the `src/` directory.

### Running the Application

Launch the Streamlit interface by running the following command in your terminal:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser to start chatting with your local AI!

---

## 📊 Evaluation

Building a RAG system is easy; proving it answers *correctly* is the hard part. This project ships a reproducible evaluation harness (`eval/`) that runs a golden question/answer set — grounded in the test document — through the exact pipeline the app uses, and scores every answer with three complementary signals so no single metric can hide a weakness.

**Latest run** — 11 questions · `all-MiniLM-L6-v2` embeddings · `phi3:mini` generator · retriever top-k = 3 · strict grounding:

| Metric | Signal | Mean (0–1) |
|--------|--------|-----------|
| Lexical overlap | ROUGE-L F1 | **0.534** |
| Semantic similarity | BERTScore F1 | **0.888** |
| Answer correctness | LLM-as-judge (`phi3:mini`) | **0.614** |
| Retrieval coverage | chunks retrieved / question | **11 / 11** |

High semantic similarity alongside a lower lexical score is the expected RAG signature: answers are *meaning-correct* but phrased differently from the reference. The set also includes an **out-of-scope question** ("What is the capital of France?") that the strictly grounded pipeline correctly **refuses** instead of hallucinating.

**What the eval surfaced:** on one in-document question the top-3 retriever missed the relevant chunk, so the model correctly refused rather than guess — a real *retrieval* gap (not a generation one) that points directly to the next improvement: a reranker or higher top-k. Catching that automatically is exactly why the harness exists.

Reproduce it (with `ollama serve` running and `phi3:mini` pulled):

```bash
python -m eval.evaluate                        # full run, writes eval/results/
python -m eval.evaluate --judge-model llama3   # use a stronger judge model
python -m eval.evaluate --no-judge             # skip the LLM judge (faster)
```

See [`eval/README.md`](eval/README.md) for methodology and per-question results.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| LLM Framework | LangChain |
| Generative Model | Phi-3 |
| Vector Database | FAISS (with on-disk index caching in `vector_store.py`) |
| Frontend / UI | Streamlit |
| Evaluation | ROUGE-L · BERTScore · LLM-as-judge |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) © 2026 Prakhar Srivastava.

---

## 🙋 Author

**Prakhar Srivastava**

Data Analyst, Data Scientist & AI Engineer | Machine Learning, Deep Learning, Generative AI, Prompt Engineering & Agentic AI
