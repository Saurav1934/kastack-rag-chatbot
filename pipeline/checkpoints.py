"""Build topic checkpoints from parsed conversation messages.

Phase implemented in this file:
1) Load `storage/messages.json`
2) Group messages by conversation (chronological order)
3) Detect topic shifts using rolling window embeddings + cosine similarity
4) Summarize each topic segment
5) Save `storage/topic_checkpoints.json`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import numpy as np
from tqdm.auto import tqdm

try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover
    hf_pipeline = None

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None


LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class CheckpointConfig:
    messages_path: Path
    output_path: Path
    output_messages_checkpoints_path: Path
    embedding_model_name: str
    message_checkpoint_size: int
    similarity_threshold: float
    window_size: int
    min_segment_messages: int
    summary_backend: str
    summary_model_name: str
    claude_model_name: str
    summary_max_length: int
    summary_min_length: int
    mode: str
    limit_conversations: int | None
    limit_message_checkpoints: int | None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_messages(messages_path: Path) -> List[Dict[str, Any]]:
    if not messages_path.exists():
        raise FileNotFoundError(f"messages.json not found: {messages_path}")

    with messages_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("messages.json must contain a JSON list.")

    required_fields = {"message_id", "conversation_id", "speaker", "text"}
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"messages[{idx}] must be an object.")
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(f"messages[{idx}] missing fields: {sorted(missing)}")

    payload.sort(key=lambda x: x["message_id"])
    return payload


def group_by_conversation(messages: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for message in messages:
        conv_id = int(message["conversation_id"])
        grouped.setdefault(conv_id, []).append(message)
    return grouped


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector
    return vector / norm


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    vec_a = normalize_vector(vec_a)
    vec_b = normalize_vector(vec_b)
    return float(np.dot(vec_a, vec_b))


def build_window_text(messages: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{item['speaker']}: {str(item['text']).strip()}" for item in messages if str(item["text"]).strip()
    )


class ClaudeSummarizer:
    """Summarize conversation segments via the Anthropic API."""

    def __init__(self, model_name: str, max_tokens: int) -> None:
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for Claude summarization. "
                "Install with: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for --summary-backend claude"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens

    def summarize(self, text: str) -> str:
        prompt = (
            "Summarize the following conversation excerpt in 2-4 sentences. "
            "Focus on the main topic, key facts, and any decisions or plans mentioned. "
            "Do not include speaker labels or quote raw dialogue.\n\n"
            f"{text[:6000]}"
        )
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = response.content[0].text.strip()
                if summary:
                    return summary
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Claude summarization attempt %s failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return extractive_fallback(text)


def extractive_fallback(text: str) -> str:
    """Last-resort fallback when no summarization backend is available."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 4:
        return text[:600]
    stitched = "\n".join([*lines[:2], "...", *lines[-2:]])
    return stitched[:900]


def create_summarizer(backend: str, model_name: str, claude_model_name: str, max_tokens: int):
    if backend == "claude":
        try:
            return ClaudeSummarizer(model_name=claude_model_name, max_tokens=max_tokens)
        except (ImportError, ValueError) as exc:
            LOGGER.warning("%s Falling back to extractive summary.", exc)
            return None

    if backend == "bart":
        if hf_pipeline is None:
            LOGGER.warning("transformers is unavailable. Using extractive fallback summarizer.")
            return None
        try:
            return hf_pipeline("summarization", model=model_name)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "Could not load summarization model '%s' (%s). Falling back to extractive summary.",
                model_name,
                exc,
            )
            return None

    LOGGER.info("Using extractive summarization backend.")
    return None


def summarize_segment(
    segment_messages: Sequence[Dict[str, Any]],
    summarizer,
    max_length: int,
    min_length: int,
) -> str:
    text = build_window_text(segment_messages)
    if not text:
        return ""

    if summarizer is None:
        return extractive_fallback(text)

    if isinstance(summarizer, ClaudeSummarizer):
        return summarizer.summarize(text)

    model_input = text[:3000]
    try:
        result = summarizer(
            model_input,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
        )
        summary = result[0]["summary_text"].strip()
        return summary if summary else extractive_fallback(model_input)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Summarization failed: %s. Using fallback extractive summary.", exc)
        return extractive_fallback(model_input)


def finalize_topic_segment(
    *,
    topic_id: int,
    conversation_id: int,
    segment_messages: Sequence[Dict[str, Any]],
    summarizer,
    summary_max_length: int,
    summary_min_length: int,
) -> Dict[str, Any]:
    return {
        "topic_id": topic_id,
        "conversation_id": conversation_id,
        "start_message": int(segment_messages[0]["message_id"]),
        "end_message": int(segment_messages[-1]["message_id"]),
        "summary": summarize_segment(
            segment_messages=segment_messages,
            summarizer=summarizer,
            max_length=summary_max_length,
            min_length=summary_min_length,
        ),
    }


def detect_topic_checkpoints(
    grouped_messages: Dict[int, List[Dict[str, Any]]],
    embedding_model: "SentenceTransformer",
    summarizer,
    similarity_threshold: float,
    window_size: int,
    min_segment_messages: int,
    summary_max_length: int,
    summary_min_length: int,
) -> List[Dict[str, Any]]:
    checkpoints: List[Dict[str, Any]] = []
    topic_id = 1

    for conversation_id in tqdm(sorted(grouped_messages.keys()), desc="Processing conversations"):
        messages = grouped_messages[conversation_id]
        if not messages:
            continue

        segment_start_idx = 0

        # A rolling comparison between "previous window" and "current window".
        for idx in range(window_size * 2, len(messages) + 1):
            previous_window = messages[idx - (window_size * 2) : idx - window_size]
            current_window = messages[idx - window_size : idx]
            prev_text = build_window_text(previous_window)
            curr_text = build_window_text(current_window)

            if not prev_text or not curr_text:
                continue

            embeddings = embedding_model.encode([prev_text, curr_text], convert_to_numpy=True)
            similarity = cosine_similarity(embeddings[0], embeddings[1])

            segment_len = idx - segment_start_idx
            if similarity < similarity_threshold and segment_len >= min_segment_messages:
                segment_messages = messages[segment_start_idx : idx - window_size]
                if segment_messages:
                    checkpoints.append(
                        finalize_topic_segment(
                            topic_id=topic_id,
                            conversation_id=conversation_id,
                            segment_messages=segment_messages,
                            summarizer=summarizer,
                            summary_max_length=summary_max_length,
                            summary_min_length=summary_min_length,
                        )
                    )
                    topic_id += 1
                segment_start_idx = idx - window_size

        # Flush remaining messages in this conversation as the final topic segment.
        remaining_segment = messages[segment_start_idx:]
        if remaining_segment:
            checkpoints.append(
                finalize_topic_segment(
                    topic_id=topic_id,
                    conversation_id=conversation_id,
                    segment_messages=remaining_segment,
                    summarizer=summarizer,
                    summary_max_length=summary_max_length,
                    summary_min_length=summary_min_length,
                )
            )
            topic_id += 1

    return checkpoints


def save_json(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def parse_args() -> CheckpointConfig:
    parser = argparse.ArgumentParser(description="Generate topic and/or message checkpoints from messages.json.")
    parser.add_argument(
        "--messages-path",
        type=Path,
        default=Path("storage/messages.json"),
        help="Path to parsed messages JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("storage/topic_checkpoints.json"),
        help="Path to output topic checkpoints JSON.",
    )
    parser.add_argument(
        "--output-messages-checkpoints-path",
        type=Path,
        default=Path("storage/message_checkpoints.json"),
        help="Path to output 100-message checkpoints JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="topic",
        choices=["topic", "messages", "both"],
        help="Which checkpoints to generate.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model.",
    )
    parser.add_argument(
        "--message-checkpoint-size",
        type=int,
        default=100,
        help="How many messages per global checkpoint (Phase 2B).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Topic split threshold; lower cosine similarity triggers topic checkpoint.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Rolling window size in messages for topic similarity checks.",
    )
    parser.add_argument(
        "--min-segment-messages",
        type=int,
        default=8,
        help="Minimum messages before allowing a split.",
    )
    parser.add_argument(
        "--summary-backend",
        type=str,
        default="claude",
        choices=["claude", "bart", "extractive"],
        help="Summarization backend: claude (API), bart (local HF), or extractive.",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default="facebook/bart-large-cnn",
        help="HuggingFace summarization model (used when --summary-backend bart).",
    )
    parser.add_argument(
        "--claude-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model name (used when --summary-backend claude).",
    )
    parser.add_argument(
        "--summary-max-length",
        type=int,
        default=120,
        help="Summarization max token length.",
    )
    parser.add_argument(
        "--summary-min-length",
        type=int,
        default=30,
        help="Summarization min token length.",
    )
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Optional: limit number of conversations processed (for faster testing).",
    )
    parser.add_argument(
        "--limit-message-checkpoints",
        type=int,
        default=None,
        help="Optional: limit number of 100-message checkpoints generated (for faster testing).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.similarity_threshold <= 1.0:
        raise ValueError("--similarity-threshold must be in [0.0, 1.0]")
    if args.window_size < 2:
        raise ValueError("--window-size must be >= 2")
    if args.min_segment_messages < args.window_size:
        raise ValueError("--min-segment-messages must be >= --window-size")
    if args.message_checkpoint_size <= 0:
        raise ValueError("--message-checkpoint-size must be > 0")

    return CheckpointConfig(
        messages_path=args.messages_path,
        output_path=args.output_path,
        output_messages_checkpoints_path=args.output_messages_checkpoints_path,
        embedding_model_name=args.embedding_model,
        message_checkpoint_size=args.message_checkpoint_size,
        similarity_threshold=args.similarity_threshold,
        window_size=args.window_size,
        min_segment_messages=args.min_segment_messages,
        summary_backend=args.summary_backend,
        summary_model_name=args.summary_model,
        claude_model_name=args.claude_model,
        summary_max_length=args.summary_max_length,
        summary_min_length=args.summary_min_length,
        mode=args.mode,
        limit_conversations=args.limit_conversations,
        limit_message_checkpoints=args.limit_message_checkpoints,
    )


def build_message_checkpoints(
    messages: Sequence[Dict[str, Any]],
    *,
    checkpoint_size: int,
    summarizer,
    summary_max_length: int,
    summary_min_length: int,
    limit_checkpoints: int | None,
) -> List[Dict[str, Any]]:
    checkpoints: List[Dict[str, Any]] = []
    total = len(messages)
    checkpoint_id = 1

    for start_idx in tqdm(range(0, total, checkpoint_size), desc="Building 100-msg checkpoints"):
        end_idx = min(total, start_idx + checkpoint_size)
        block = messages[start_idx:end_idx]
        if not block:
            continue

        checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "start_message": int(block[0]["message_id"]),
                "end_message": int(block[-1]["message_id"]),
                "summary": summarize_segment(
                    segment_messages=block,
                    summarizer=summarizer,
                    max_length=summary_max_length,
                    min_length=summary_min_length,
                ),
            }
        )
        checkpoint_id += 1

        if limit_checkpoints is not None and len(checkpoints) >= limit_checkpoints:
            break

    return checkpoints


def main() -> None:
    setup_logging()
    config = parse_args()

    LOGGER.info("Loading messages from %s", config.messages_path)
    messages = load_messages(config.messages_path)
    LOGGER.info("Loaded %s messages", len(messages))

    summarizer = create_summarizer(
        backend=config.summary_backend,
        model_name=config.summary_model_name,
        claude_model_name=config.claude_model_name,
        max_tokens=config.summary_max_length,
    )

    if config.mode in ("topic", "both"):
        grouped_messages = group_by_conversation(messages)
        LOGGER.info("Grouped into %s conversations", len(grouped_messages))

        if config.limit_conversations is not None:
            limited_keys = sorted(grouped_messages.keys())[: config.limit_conversations]
            grouped_messages = {k: grouped_messages[k] for k in limited_keys}
            LOGGER.info("Limited to %s conversations for testing", len(grouped_messages))

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for topic checkpoints. "
                "Install with: pip install sentence-transformers"
            ) from exc

        LOGGER.info("Loading embedding model: %s", config.embedding_model_name)
        embedding_model = SentenceTransformer(config.embedding_model_name)

        LOGGER.info("Detecting topic checkpoints...")
        checkpoints = detect_topic_checkpoints(
            grouped_messages=grouped_messages,
            embedding_model=embedding_model,
            summarizer=summarizer,
            similarity_threshold=config.similarity_threshold,
            window_size=config.window_size,
            min_segment_messages=config.min_segment_messages,
            summary_max_length=config.summary_max_length,
            summary_min_length=config.summary_min_length,
        )
        LOGGER.info("Generated %s topic checkpoints", len(checkpoints))

        save_json(checkpoints, config.output_path)
        LOGGER.info("Saved topic checkpoints to %s", config.output_path)

    if config.mode in ("messages", "both"):
        LOGGER.info(
            "Generating message checkpoints (every %s messages)",
            config.message_checkpoint_size,
        )
        message_checkpoints = build_message_checkpoints(
            messages,
            checkpoint_size=config.message_checkpoint_size,
            summarizer=summarizer,
            summary_max_length=config.summary_max_length,
            summary_min_length=config.summary_min_length,
            limit_checkpoints=config.limit_message_checkpoints,
        )
        LOGGER.info("Generated %s message checkpoints", len(message_checkpoints))
        save_json(message_checkpoints, config.output_messages_checkpoints_path)
        LOGGER.info(
            "Saved message checkpoints to %s",
            config.output_messages_checkpoints_path,
        )


if __name__ == "__main__":
    main()