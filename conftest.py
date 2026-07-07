"""Ensures the project root is importable so tests can `import src...`.

`src` and `eval` are namespace packages (no __init__.py), imported relative to
the repo root the same way `streamlit run app.py` and `python -m eval.evaluate`
do. Adding the root to sys.path makes that work under pytest too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
