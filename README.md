# KaStack Conversation Memory RAG

A retrieval-augmented chatbot that answers questions about a user's habits, personality, and conversation history from ~11,000 synthetic chat conversations (~191K messages).

## Architecture

```
data/conversations.csv
       │
       ▼
pipeline/parse.py  ──►  storage/messages.json
       │
       ├── pipeline/checkpoints.py  ──►  topic_checkpoints.json
       │                            └──  message_checkpoints.json
       │
       ├── pipeline/retrieval.py    ──►  storage/faiss_index/  (3 FAISS indexes)
       │
       └── pipeline/persona.py     ──►  storage/persona.json
                                              │
                                              ▼
                                    chatbot/app.py  (Streamlit)
```

## Topic Detection

Topic checkpoints are built **per conversation**, in chronological order:

1. Messages are grouped by `conversation_id` and sorted by `message_id`.
2. A rolling window (default 5 messages) slides through each conversation.
3. Adjacent windows are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
4. When cosine similarity between consecutive windows drops below **0.5**, a topic boundary is created (minimum 8 messages per segment).
5. Each segment is summarized with the **Claude API** (default) into 2–4 sentences.

This ensures topic splits respect conversation boundaries and chronological order — a requirement for correct evaluation.

## Retrieval

Three FAISS indexes (`IndexFlatIP` on normalized embeddings):

| Index | Source | Use |
|-------|--------|-----|
| `topic_summaries.index` | Topic checkpoint summaries | High-level topic recall |
| `message_checkpoints.index` | 100-message block summaries | Mid-range context |
| `message_chunks.index` | Raw 100-message text blocks | Exact quote retrieval |

Query flow: embed query → search all 3 indexes (top-k) → combine with persona data → generate grounded answer.

## Persona Extraction

Profiles **User 1 only** across all conversations. The CSV uses generic `User 1` / `User 2` speaker labels per conversation — these are not the same person across conversations. We treat **User 1 as the primary speaker** in each thread and aggregate patterns (habits, facts, traits, communication style) with regex rules backed by explicit message evidence.

Extracted fields:
- **Habits** — coffee drinker, runner, home cook, etc.
- **Personal facts** — relationship status, location mentions, career
- **Personality traits** — enthusiastic, humorous, ambitious, etc.
- **Communication style** — tone, message length, emoji usage, punctuation

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Parse conversations (if messages.json is missing)

```bash
python pipeline/parse.py
```

### Build checkpoints

Set your Anthropic API key for real summaries:

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-key-here"

# macOS/Linux
export ANTHROPIC_API_KEY="your-key-here"
```

**Test run** (500 conversations):

```bash
python pipeline/checkpoints.py --mode both --limit-conversations 500
```

**Full run** (all 11,001 conversations — slow, many API calls):

```bash
python pipeline/checkpoints.py --mode both
```

Options:
- `--summary-backend claude` (default) — Claude API summaries
- `--summary-backend bart` — local `facebook/bart-large-cnn` (needs GPU/RAM)
- `--summary-backend extractive` — fast fallback, not real summaries

### Build FAISS indexes

```bash
python pipeline/retrieval.py
```

### Extract persona

```bash
python pipeline/persona.py
```

### Run chatbot locally

```bash
streamlit run chatbot/app.py
```

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub (exclude `.venv/` and `storage/messages.json` — already in `.gitignore`).
2. Commit the built artifacts: `storage/topic_checkpoints.json`, `storage/message_checkpoints.json`, `storage/persona.json`, and `storage/faiss_index/`.
3. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
4. Set **Main file path** to `chatbot/app.py`.
5. Deploy.

For a live demo URL, add it here after deployment: `https://your-app.streamlit.app`

## Project Structure

```
kastack-rag/
├── chatbot/
│   └── app.py              # Streamlit chatbot UI
├── data/
│   └── conversations.csv   # Source dataset
├── pipeline/
│   ├── parse.py            # CSV → messages.json
│   ├── checkpoints.py      # Topic + message checkpoints
│   ├── retrieval.py        # FAISS index builder + retriever
│   └── persona.py          # User 1 persona extraction
├── storage/
│   ├── messages.json       # Parsed messages (gitignored, 30MB)
│   ├── topic_checkpoints.json
│   ├── message_checkpoints.json
│   ├── persona.json
│   └── faiss_index/
├── requirements.txt
└── README.md
```

## Assumptions

- **User 1 persona**: `User 1` and `User 2` are generic labels reused across independent conversations. Persona extraction targets `User 1` as the profiled speaker in each thread.
- **Chronological ordering**: All messages are globally sorted by `message_id`; topic detection runs within each `conversation_id` group.
- **Summarization**: Claude API is the default backend. Without `ANTHROPIC_API_KEY`, the pipeline falls back to extractive text (not suitable for submission).
