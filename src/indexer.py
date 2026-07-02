"""
Layer 2B: Internal Library Indexer.

Chunks the internal library, embeds each chunk, and builds a FAISS
IndexFlatIP (cosine similarity via inner product on L2-normalized vectors).

Embedder resolution order:
  1. SentenceTransformerEmbedder (all-MiniLM-L6-v2) -- used if the package
     is installed AND the model weights can be loaded (needs internet on
     first run, or a pre-downloaded local cache).
  2. HashingEmbedder -- scikit-learn HashingVectorizer + TruncatedSVD.
     Deterministic, dependency-light, fully offline. Used automatically if
     (1) fails for any reason, or if FORCE_HASHING_EMBEDDER=true.

This fallback is what lets the whole system run with zero internet access
and zero GPU.
"""
import logging
import pickle
from typing import List, Dict, Tuple

import numpy as np

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder abstraction
# ---------------------------------------------------------------------------
class Embedder:
    """Common interface: encode(texts) -> L2-normalized float32 array."""

    name = "base"
    dim = None

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer  # may raise

        model_name = model_name or config.SENTENCE_TRANSFORMER_MODEL
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        vecs = vecs.astype("float32")
        return _l2_normalize(vecs)


class HashingEmbedder(Embedder):
    """Deterministic, offline-safe fallback embedder.

    HashingVectorizer -> TruncatedSVD gives dense, fixed-dimension vectors
    without ever downloading anything or requiring a fitted vocabulary.
    Semantic quality is lower than a real sentence embedding model, but it
    is stable, fast, and never fails -- which is the point of a fallback.

    IMPORTANT: TruncatedSVD must be fit ONCE on a representative corpus
    (via `fit()`, called during index build) and then reused for every
    later `encode()` call, including single-text query encoding at search
    time. Refitting per call would produce a different, incompatible
    vector space on each call -- that was a real bug caught during testing
    (query-time encode() was silently refitting on a 1-sample batch,
    producing a dimension mismatch against the saved FAISS index).
    """

    name = "hashing-svd"

    def __init__(self, dim: int = None):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim or config.EMBEDDING_DIM_FALLBACK
        self._vectorizer = HashingVectorizer(
            n_features=2 ** 14, alternate_sign=False, norm=None
        )
        self._svd = None
        self._svd_fitted = False

    def fit(self, texts: List[str]):
        """Fit the SVD projection once, on the full corpus used to build
        the index. Must be called before encode() is used for queries."""
        from sklearn.decomposition import TruncatedSVD

        sparse = self._vectorizer.transform(texts)
        n_components = min(self.dim, max(2, min(sparse.shape) - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(sparse)
        self.dim = n_components
        self._svd_fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        sparse = self._vectorizer.transform(texts)
        if not self._svd_fitted:
            # Safety net: encode() called before fit() (e.g. a caller using
            # this class directly rather than through LibraryIndex.build).
            # Fit on whatever batch was given so the call never crashes,
            # but this should not happen in the normal index -> search flow.
            logger.warning(
                "HashingEmbedder.encode() called before fit(); "
                "fitting on this batch as a fallback (dimension may vary)."
            )
            self.fit(texts)
        dense = self._svd.transform(sparse)
        dense = dense.astype("float32")
        return _l2_normalize(dense)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def get_embedder() -> Embedder:
    """Factory: try sentence-transformers, fall back to hashing embedder."""
    if config.FORCE_HASHING_EMBEDDER:
        logger.info("FORCE_HASHING_EMBEDDER=true -> using HashingEmbedder")
        return HashingEmbedder()
    try:
        embedder = SentenceTransformerEmbedder()
        logger.info("Using SentenceTransformerEmbedder (%s)", config.SENTENCE_TRANSFORMER_MODEL)
        return embedder
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for graceful fallback
        logger.warning(
            "sentence-transformers unavailable (%s); falling back to HashingEmbedder",
            exc,
        )
        return HashingEmbedder()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_words: int = 100) -> List[str]:
    """Simple word-count chunker. Our synthetic articles are short (one
    chunk each in practice), but this scales to longer real-world text."""
    words = text.split()
    if not words:
        return [""]
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


# ---------------------------------------------------------------------------
# Library index
# ---------------------------------------------------------------------------
class LibraryIndex:
    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder or get_embedder()
        self.index = None  # faiss.IndexFlatIP
        self.metadata: List[Dict] = []  # position -> chunk metadata

    def build(self, library_df) -> "LibraryIndex":
        import faiss

        chunk_texts = []
        self.metadata = []
        for _, row in library_df.iterrows():
            chunks = chunk_text(row["full_text"])
            for chunk in chunks:
                chunk_texts.append(chunk)
                self.metadata.append(
                    {
                        "article_id": row["article_id"],
                        "headline": row["headline"],
                        "chunk_text": chunk,
                        "category": row["category"],
                        "seed_topic": row["seed_topic"],
                        "date": row["date"],
                    }
                )

        # HashingEmbedder needs an explicit corpus fit before it can encode
        # queries consistently (see HashingEmbedder docstring). Other
        # embedders (e.g. sentence-transformers) don't need this and
        # silently ignore it via getattr.
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(chunk_texts)

        vectors = self.embedder.encode(chunk_texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        logger.info(
            "Built FAISS index: %d chunks, dim=%d, embedder=%s",
            len(chunk_texts), dim, self.embedder.name,
        )
        return self

    def search(self, query: str, k: int = None) -> List[Tuple[Dict, float]]:
        k = k or config.RAG_TOP_K
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        qvec = self.embedder.encode([query])
        scores, idxs = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, index_path=None, metadata_path=None):
        import faiss

        index_path = index_path or config.FAISS_INDEX_PATH
        metadata_path = metadata_path or config.METADATA_PATH
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            # Persist the embedder object itself (not just its name). This
            # matters for HashingEmbedder, whose SVD projection is fit on
            # the corpus at build time -- reloading must reuse that exact
            # fitted transform, not a fresh unfitted embedder, or query
            # vectors won't be compatible with the saved FAISS index.
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "embedder_name": self.embedder.name,
                    "embedder": self.embedder,
                },
                f,
            )
        logger.info("Saved index -> %s, metadata -> %s", index_path, metadata_path)

    @classmethod
    def load(cls, index_path=None, metadata_path=None, embedder: Embedder = None) -> "LibraryIndex":
        import faiss

        index_path = index_path or config.FAISS_INDEX_PATH
        metadata_path = metadata_path or config.METADATA_PATH
        with open(metadata_path, "rb") as f:
            payload = pickle.load(f)

        # Prefer the caller-supplied embedder (e.g. tests forcing a specific
        # backend); otherwise reuse the exact fitted embedder that was
        # saved alongside this index.
        resolved_embedder = embedder or payload.get("embedder") or get_embedder()

        instance = cls(embedder=resolved_embedder)
        instance.index = faiss.read_index(str(index_path))
        instance.metadata = payload["metadata"]
        return instance


def build_index(library_df, embedder: Embedder = None) -> LibraryIndex:
    return LibraryIndex(embedder=embedder).build(library_df)


def load_index(embedder: Embedder = None) -> LibraryIndex:
    return LibraryIndex.load(embedder=embedder)
