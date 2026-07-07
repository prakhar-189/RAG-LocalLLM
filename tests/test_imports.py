"""Smoke tests: the backend modules import cleanly and app.py compiles."""
import importlib
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_src_modules_import():
    for module in ("src.vector_store", "src.rag_pipeline", "src.llm_model"):
        importlib.import_module(module)


def test_app_compiles():
    # app.py runs Streamlit calls at import time, so only byte-compile it.
    py_compile.compile(str(ROOT / "app.py"), doraise=True)
