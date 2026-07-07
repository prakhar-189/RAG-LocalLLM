# vector_store.py
# ----------------------------------------------
# This module is responsible for turning raw documents into searchable math.
# Loads a PDF document using LangChain's PyPDFLoader.
# Then applies advanced semantic chunking to maintain contextual integrity.
# Converts those text chunks into numerical vectors using HuggingFace's sentence-transformers model.
# Then creates a FAISS vector store for efficient retrieval.
#
# Persistence:
#   Building the store is the expensive step -- SemanticChunker embeds every
#   sentence in the document, so re-processing the same PDF on every upload
#   wastes minutes of CPU. We fingerprint the file by its content hash and
#   cache the FAISS index on disk. A second run on the same document loads the
#   index instantly instead of re-embedding it.
# ----------------------------------------------


# =========================================
# Libraries
# ------------------------------------
# PyPDFLoader : For loading PDF documents.
# HuggingFaceEmbeddings : For generating embeddings using HuggingFace models.
# SemanticChunker : For advanced text splitting based on semantic similarity.
# FAISS : For creating a vector store for efficient similarity search.
# ========================================
import hashlib
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Embedding model + cache location are configurable via env vars.
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CACHE_DIR = Path(os.environ.get("VECTOR_CACHE_DIR", ".vector_cache"))


def _load_embeddings():
    """Returns the HuggingFace embedding model used for chunking and retrieval."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _fingerprint(file_path):
    """
    Builds a stable cache key from the file's *contents* (not its name/path).

    Two uploads of the same PDF -- even from different temp paths -- produce the
    same fingerprint and therefore reuse the same cached index. The embedding
    model name is folded in so the cache invalidates automatically if the model
    ever changes.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            sha.update(block)
    sha.update(EMBEDDING_MODEL_NAME.encode("utf-8"))
    return sha.hexdigest()[:16]


# =========================================
# create_vector_store Function
# ------------------------------------
# Takes a path to a PDF, loads it, applies semantic chunking to maintain
# contextual integrity, and returns a FAISS vector store for efficient retrieval.
# When use_cache is True (default), a prebuilt index for the same document is
# loaded from disk if present; otherwise the freshly built index is saved for
# next time.
# =========================================
def create_vector_store(file_path, use_cache=True):
    """Loads a PDF, applies semantic chunking, and returns a FAISS vector store."""

    embeddings = _load_embeddings()

    # Fast path: reuse a previously built index for this exact document.
    if use_cache:
        cache_path = CACHE_DIR / _fingerprint(file_path)
        if cache_path.exists():
            # allow_dangerous_deserialization is required by FAISS to unpickle a
            # local index. Safe here: we only ever load indexes we created.
            return FAISS.load_local(
                str(cache_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

    # Load the PDF file from the provided path
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Apply advanced semantic chunking (splits on meaning, not fixed length)
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
    )
    chunks = text_splitter.split_documents(documents)

    # Create the FAISS vector store from the embedded chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist the built index so the next run on this document is instant.
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_path))

    return vectorstore
