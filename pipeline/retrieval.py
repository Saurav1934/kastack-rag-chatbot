"""FAISS retrieval system (Phase 2C).

Builds 3 FAISS indexes:
1) Topic summaries (from storage/topic_checkpoints.json)
2) Message checkpoint summaries (from storage/message_checkpoints.json)
3) Raw message chunks (100-message blocks from storage/messages.json)

Indexes + metadata are stored in `storage/faiss_index/`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """Normalize to unit length so inner product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def build_message_chunks(
    messages: Sequence[Dict[str, Any]],
    *,
    chunk_size: int = 100,
    text_limit_chars: int = 7000,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Create raw message chunks for embedding + retrieval."""
    texts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    chunk_id = 1

    for start_idx in tqdm(range(0, len(messages), chunk_size), desc="Building raw message chunks"):
        end_idx = min(len(messages), start_idx + chunk_size)
        block = messages[start_idx:end_idx]
        if not block:
            continue

        # Keep embeddings inputs and stored text reasonably bounded.
        lines = []
        for m in block:
            speaker = str(m.get("speaker", "")).strip()
            text = str(m.get("text", "")).strip()
            if text:
                lines.append(f"{speaker}: {text}".strip())

        chunk_text = "\n".join(lines).strip()
        if text_limit_chars is not None and len(chunk_text) > text_limit_chars:
            chunk_text = chunk_text[:text_limit_chars]

        texts.append(chunk_text)
        metadata.append(
            {
                "chunk_id": chunk_id,
                "start_message": int(block[0]["message_id"]),
                "end_message": int(block[-1]["message_id"]),
                "text": chunk_text,
            }
        )
        chunk_id += 1

    return texts, metadata


def create_faiss_index(embeddings: np.ndarray):
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(embeddings)
    return index


def encode_texts(embedding_model, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
    """Embed in batches. Returns float32 numpy array with shape [n, dim]."""
    # sentence-transformers supports normalize_embeddings, but to keep it explicit
    # we normalize ourselves so cosine math is unambiguous across environments.
    vectors = embedding_model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    vectors = normalize_embeddings(np.asarray(vectors))
    return vectors.astype(np.float32)


@dataclass(frozen=True)
class RetrievalBuildConfig:
    messages_path: Path
    topic_checkpoints_path: Path
    message_checkpoints_path: Path
    index_dir: Path
    embedding_model_name: str
    batch_size: int
    top_k: int
    chunk_size: int
    chunk_text_limit_chars: int
    limit_topics: Optional[int]
    limit_message_checkpoints: Optional[int]
    limit_message_chunks: Optional[int]


def parse_args() -> RetrievalBuildConfig:
    parser = argparse.ArgumentParser(description="Build FAISS indexes for RAG retrieval.")
    parser.add_argument("--messages-path", type=Path, default=Path("storage/messages.json"))
    parser.add_argument(
        "--topic-checkpoints-path", type=Path, default=Path("storage/topic_checkpoints.json")
    )
    parser.add_argument(
        "--message-checkpoints-path", type=Path, default=Path("storage/message_checkpoints.json")
    )
    parser.add_argument("--index-dir", type=Path, default=Path("storage/faiss_index"))
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--chunk-text-limit-chars", type=int, default=7000)
    parser.add_argument("--limit-topics", type=int, default=None)
    parser.add_argument("--limit-message-checkpoints", type=int, default=None)
    parser.add_argument("--limit-message-chunks", type=int, default=None)

    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    return RetrievalBuildConfig(
        messages_path=args.messages_path,
        topic_checkpoints_path=args.topic_checkpoints_path,
        message_checkpoints_path=args.message_checkpoints_path,
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_text_limit_chars=args.chunk_text_limit_chars,
        limit_topics=args.limit_topics,
        limit_message_checkpoints=args.limit_message_checkpoints,
        limit_message_chunks=args.limit_message_chunks,
    )


def build_all_indexes(cfg: RetrievalBuildConfig) -> None:
    for p in [cfg.messages_path, cfg.topic_checkpoints_path, cfg.message_checkpoints_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    LOGGER.info("Loading messages...")
    messages: List[Dict[str, Any]] = load_json(cfg.messages_path)

    LOGGER.info("Loading topic checkpoints...")
    topic_checkpoints: List[Dict[str, Any]] = load_json(cfg.topic_checkpoints_path)
    if cfg.limit_topics is not None:
        topic_checkpoints = topic_checkpoints[: cfg.limit_topics]

    LOGGER.info("Loading message checkpoints...")
    message_checkpoints: List[Dict[str, Any]] = load_json(cfg.message_checkpoints_path)
    if cfg.limit_message_checkpoints is not None:
        message_checkpoints = message_checkpoints[: cfg.limit_message_checkpoints]

    LOGGER.info("Loading embedding model: %s", cfg.embedding_model_name)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers") from exc

    embedding_model = SentenceTransformer(cfg.embedding_model_name)

    cfg.index_dir.mkdir(parents=True, exist_ok=True)

    # 1) Topic summaries index
    topic_texts = [str(item.get("summary", "")).strip() for item in topic_checkpoints]
    topic_texts = [t for t in topic_texts if t]
    # Keep metadata aligned with embedded texts:
    topic_metadata = [
        item for item in topic_checkpoints if str(item.get("summary", "")).strip()
    ]
    if topic_texts:
        LOGGER.info("Embedding %s topic summaries...", len(topic_texts))
        topic_vecs = encode_texts(embedding_model, topic_texts, batch_size=cfg.batch_size)
        topic_index = create_faiss_index(topic_vecs)
        import faiss  # type: ignore

        faiss.write_index(topic_index, str(cfg.index_dir / "topic_summaries.index"))
        save_json(topic_metadata, cfg.index_dir / "topic_summaries_metadata.json")
        LOGGER.info("Saved topic index.")
    else:
        LOGGER.warning("No topic summaries found; skipping topic index build.")

    # 2) Message checkpoint summaries index
    msg_texts = [str(item.get("summary", "")).strip() for item in message_checkpoints]
    msg_texts = [t for t in msg_texts if t]
    msg_metadata = [item for item in message_checkpoints if str(item.get("summary", "")).strip()]
    if msg_texts:
        LOGGER.info("Embedding %s message checkpoint summaries...", len(msg_texts))
        msg_vecs = encode_texts(embedding_model, msg_texts, batch_size=cfg.batch_size)
        msg_index = create_faiss_index(msg_vecs)
        import faiss  # type: ignore

        faiss.write_index(msg_index, str(cfg.index_dir / "message_checkpoints.index"))
        save_json(msg_metadata, cfg.index_dir / "message_checkpoints_metadata.json")
        LOGGER.info("Saved message checkpoint index.")
    else:
        LOGGER.warning("No message checkpoint summaries found; skipping message checkpoint index build.")

    # 3) Raw message chunks index
    LOGGER.info("Building raw chunk dataset from messages.json...")
    chunk_texts, chunk_metadata = build_message_chunks(
        messages,
        chunk_size=cfg.chunk_size,
        text_limit_chars=cfg.chunk_text_limit_chars,
    )
    if cfg.limit_message_chunks is not None:
        chunk_texts = chunk_texts[: cfg.limit_message_chunks]
        chunk_metadata = chunk_metadata[: cfg.limit_message_chunks]

    if chunk_texts:
        LOGGER.info("Embedding %s raw message chunks...", len(chunk_texts))
        chunk_vecs = encode_texts(embedding_model, chunk_texts, batch_size=cfg.batch_size)
        chunk_index = create_faiss_index(chunk_vecs)
        import faiss  # type: ignore

        faiss.write_index(chunk_index, str(cfg.index_dir / "message_chunks.index"))
        save_json(chunk_metadata, cfg.index_dir / "message_chunks_metadata.json")
        LOGGER.info("Saved raw message chunk index.")
    else:
        LOGGER.warning("No raw chunks found; skipping message chunk index build.")


def _load_faiss_index(path: Path):
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss.read_index(str(path))


class FaissRetriever:
    """Load FAISS indexes + metadata and retrieve top-k matches for a query."""

    def __init__(
        self,
        *,
        index_dir: Path,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            ) from exc

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index_dir = index_dir

        self.topic_index = _load_faiss_index(index_dir / "topic_summaries.index")
        self.topic_meta = load_json(index_dir / "topic_summaries_metadata.json")

        self.msg_index = _load_faiss_index(index_dir / "message_checkpoints.index")
        self.msg_meta = load_json(index_dir / "message_checkpoints_metadata.json")

        self.chunk_index = _load_faiss_index(index_dir / "message_chunks.index")
        self.chunk_meta = load_json(index_dir / "message_chunks_metadata.json")

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embedding_model.encode([query], convert_to_numpy=True)
        vec = normalize_embeddings(np.asarray(vec))
        return vec.astype(np.float32)

    def search(self, query: str, *, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        query_vec = self.embed_query(query)

        # inner product with normalized vectors => cosine similarity
        topic_scores, topic_ids = self.topic_index.search(query_vec, top_k)
        msg_scores, msg_ids = self.msg_index.search(query_vec, top_k)
        chunk_scores, chunk_ids = self.chunk_index.search(query_vec, top_k)

        def pack(meta: List[Dict[str, Any]], ids: np.ndarray, scores: np.ndarray) -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []
            for rank in range(len(ids[0])):
                idx = int(ids[0][rank])
                if idx < 0:
                    continue
                results.append(
                    {
                        "score": float(scores[0][rank]),
                        **meta[idx],
                    }
                )
            return results

        return {
            "topic_summaries": pack(self.topic_meta, topic_ids, topic_scores),
            "message_checkpoints": pack(self.msg_meta, msg_ids, msg_scores),
            "message_chunks": pack(self.chunk_meta, chunk_ids, chunk_scores),
        }


def main() -> None:
    setup_logging()
    cfg = parse_args()
    build_all_indexes(cfg)


if __name__ == "__main__":
    main()

