# src/llm_model.py
# ------------------------------------------------
# This module initializes the AI model used in the application.
# It defines a function to load the local Ollama model (Phi-3) using LangChain's Ollama wrapper.
# ------------------------------------------------


# =========================================
# Libraries
# ------------------------------------
# OllamaLLM : Current LangChain integration for local Ollama models. This
#   replaces langchain_community.llms.Ollama, which is deprecated in favor of
#   the dedicated langchain-ollama package.
# =========================================
import os

from langchain_ollama import OllamaLLM

# Model + host are configurable via env vars (defaults keep it fully local).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")  # e.g. http://host.docker.internal:11434


def load_llm():
    """Initializes and returns the local Ollama model."""
    kwargs = {"model": OLLAMA_MODEL, "temperature": 0.0}
    if OLLAMA_BASE_URL:
        kwargs["base_url"] = OLLAMA_BASE_URL
    return OllamaLLM(**kwargs)