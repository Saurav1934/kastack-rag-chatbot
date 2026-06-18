"""Streamlit chatbot for conversation memory RAG (Phase 4).

Retrieves relevant topic checkpoints, message checkpoints, raw message chunks,
and persona data to produce grounded answers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.retrieval import FaissRetriever, load_json  # noqa: E402

STORAGE_DIR = ROOT / "storage"
PERSONA_PATH = STORAGE_DIR / "persona.json"
INDEX_DIR = STORAGE_DIR / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLE_QUESTIONS = [
    "What kind of person is this user?",
    "What are their habits?",
    "How do they communicate?",
    "What did the user say about moving to Portland?",
    "What happened in conversations about cooking?",
]


def detect_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ("habit", "routine", "usually", "often")):
        return "habits"
    if any(k in q for k in ("communicat", "tone", "emoji", "message length", "writing style")):
        return "communication"
    if any(k in q for k in ("personality", "person", "trait", "character", "like as a person")):
        return "personality"
    if any(k in q for k in ("said about", "mention", "talk about", "discuss")):
        return "mention"
    if any(k in q for k in ("what happened", "during topic", "conversation about", "topic")):
        return "topic"
    return "general"


def format_evidence_list(items: List[Dict[str, Any]], *, key: str = "text") -> List[str]:
    lines: List[str] = []
    for item in items:
        text = str(item.get(key, "")).strip()
        if not text:
            continue
        msg_id = item.get("message_id")
        prefix = f"[msg {msg_id}] " if msg_id is not None else ""
        lines.append(f"{prefix}{text}")
    return lines


def persona_section(persona: Dict[str, Any], section: str) -> List[Dict[str, Any]]:
    return list(persona.get(section, []))


def answer_from_persona(intent: str, persona: Dict[str, Any]) -> str:
    speaker = persona.get("target_speaker", "User 1")
    style = persona.get("communication_style", {})

    if intent == "habits":
        habits = persona_section(persona, "habits")
        if not habits:
            return f"I could not find supported habit evidence for {speaker}."
        lines = [f"Based on conversation evidence, {speaker}'s habits include:"]
        for habit in habits[:8]:
            label = habit.get("label", "unknown")
            evidence = format_evidence_list(habit.get("evidence", []))
            lines.append(f"- **{label}**")
            for ev in evidence[:2]:
                lines.append(f"  - {ev}")
        return "\n".join(lines)

    if intent == "communication":
        if not style:
            return f"No communication-style profile is available for {speaker}."
        return (
            f"{speaker} communicates with a **{style.get('tone', 'unknown')}** tone, "
            f"**{style.get('avg_length_category', 'unknown')}** message length "
            f"(avg {style.get('avg_message_length', 'n/a')} chars), "
            f"{'uses' if style.get('emoji_usage') else 'rarely uses'} emojis, "
            f"and a **{style.get('punctuation_style', 'unknown')}** punctuation style."
        )

    if intent == "personality":
        traits = persona_section(persona, "personality_traits")
        facts = persona_section(persona, "personal_facts")
        if not traits and not facts:
            return f"I could not find supported personality evidence for {speaker}."
        lines = [f"From supported evidence, {speaker} appears to be:"]
        for trait in traits[:6]:
            lines.append(f"- **{trait.get('label', 'unknown')}**")
            for ev in format_evidence_list(trait.get("evidence", []))[:1]:
                lines.append(f"  - {ev}")
        if facts:
            lines.append("\nNotable personal facts:")
            for fact in facts[:5]:
                lines.append(f"- **{fact.get('label', 'unknown')}**")
        return "\n".join(lines)

    # general persona overview
    habits = persona_section(persona, "habits")[:3]
    traits = persona_section(persona, "personality_traits")[:3]
    facts = persona_section(persona, "personal_facts")[:3]
    lines = [f"Here is a grounded profile of **{speaker}**:"]
    if habits:
        lines.append("\n**Habits:** " + ", ".join(h.get("label", "") for h in habits))
    if traits:
        lines.append("**Personality:** " + ", ".join(t.get("label", "") for t in traits))
    if facts:
        lines.append("**Personal facts:** " + ", ".join(f.get("label", "") for f in facts))
    if style:
        lines.append(
            f"**Communication:** {style.get('tone', 'unknown')} tone, "
            f"{style.get('avg_length_category', 'unknown')} messages."
        )
    return "\n".join(lines)


def answer_from_retrieval(
    query: str,
    retrieval: Dict[str, List[Dict[str, Any]]],
    persona: Dict[str, Any],
) -> str:
    topics = retrieval.get("topic_summaries", [])
    checkpoints = retrieval.get("message_checkpoints", [])
    chunks = retrieval.get("message_chunks", [])

    speaker = persona.get("target_speaker", "User 1")
    lines = [f"Grounded answer for: _{query}_", ""]

    if topics:
        lines.append("**Relevant topic segments:**")
        for item in topics[:3]:
            summary = str(item.get("summary", "")).strip()
            if summary:
                lines.append(
                    f"- Topic {item.get('topic_id')} "
                    f"(msgs {item.get('start_message')}-{item.get('end_message')}): {summary}"
                )

    if checkpoints:
        lines.append("\n**Relevant message-checkpoint summaries:**")
        for item in checkpoints[:2]:
            summary = str(item.get("summary", "")).strip()
            if summary:
                lines.append(
                    f"- Checkpoint {item.get('checkpoint_id')} "
                    f"(msgs {item.get('start_message')}-{item.get('end_message')}): {summary[:400]}"
                )

    if chunks:
        lines.append(f"\n**What {speaker} said (from retrieved message chunks):**")
        for item in chunks[:2]:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            # Pull User 1 lines that seem relevant to the query keywords.
            keywords = [w for w in re.findall(r"[a-zA-Z]{4,}", query.lower()) if len(w) > 3]
            relevant_lines = []
            for line in text.splitlines():
                if not line.startswith(f"{speaker}:"):
                    continue
                line_lower = line.lower()
                if not keywords or any(k in line_lower for k in keywords):
                    relevant_lines.append(line.strip())
            shown = relevant_lines[:4] if relevant_lines else text.splitlines()[:4]
            for line in shown:
                lines.append(f"- {line[:300]}")

    if len(lines) <= 2:
        return (
            "I could not find enough retrieved context to answer that question. "
            "Try rebuilding indexes with `python pipeline/retrieval.py`."
        )

    return "\n".join(lines)


def generate_answer(
    query: str,
    persona: Dict[str, Any],
    retriever: Optional[FaissRetriever],
    *,
    top_k: int,
) -> str:
    intent = detect_intent(query)

    if intent in {"habits", "communication", "personality"}:
        persona_answer = answer_from_persona(intent, persona)
        if retriever is None:
            return persona_answer

        retrieval = retriever.search(query, top_k=top_k)
        retrieval_answer = answer_from_retrieval(query, retrieval, persona)
        return persona_answer + "\n\n---\n\n" + retrieval_answer

    if intent == "general" and any(
        k in query.lower() for k in ("kind of person", "who is", "profile", "about this user")
    ):
        return answer_from_persona("personality", persona)

    if retriever is None:
        return (
            "Retrieval indexes are not available. Persona-only questions work, "
            "but topic/message questions need FAISS indexes."
        )

    retrieval = retriever.search(query, top_k=top_k)
    return answer_from_retrieval(query, retrieval, persona)


@st.cache_resource(show_spinner="Loading retrieval indexes...")
def load_retriever() -> Optional[FaissRetriever]:
    required = [
        INDEX_DIR / "topic_summaries.index",
        INDEX_DIR / "message_checkpoints.index",
        INDEX_DIR / "message_chunks.index",
    ]
    if not all(path.exists() for path in required):
        return None
    return FaissRetriever(index_dir=INDEX_DIR, embedding_model_name=EMBEDDING_MODEL)


@st.cache_data(show_spinner=False)
def load_persona() -> Dict[str, Any]:
    if not PERSONA_PATH.exists():
        return {}
    return load_json(PERSONA_PATH)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(page_title="KaStack Conversation Memory", page_icon="💬", layout="wide")
    init_session_state()

    st.title("KaStack Conversation Memory Chatbot")
    st.caption("Grounded answers from topic checkpoints, message checkpoints, chunks, and persona.")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-K retrieval results", min_value=1, max_value=10, value=5)
        st.markdown("### Sample questions")
        for question in SAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question

        persona = load_persona()
        retriever = load_retriever()
        st.markdown("---")
        st.markdown(f"**Persona loaded:** {'Yes' if persona else 'No'}")
        st.markdown(f"**FAISS indexes loaded:** {'Yes' if retriever else 'No'}")

    persona = load_persona()
    retriever = load_retriever()

    if not persona:
        st.warning("`storage/persona.json` not found. Run `python pipeline/persona.py` first.")
    if retriever is None:
        st.info(
            "FAISS indexes not found or incomplete. Run:\n\n"
            "`python pipeline/checkpoints.py --mode both`\n\n"
            "`python pipeline/retrieval.py`"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state.pop("pending_question", None)
    user_query = st.chat_input("Ask about this user's habits, personality, or conversations...")
    if pending:
        user_query = pending

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                answer = generate_answer(
                    user_query,
                    persona=persona,
                    retriever=retriever,
                    top_k=top_k,
                )
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
