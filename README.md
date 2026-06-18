## Assignment Requirement Mapping

### Topic Checkpoints

* Conversations are processed chronologically.
* Topic changes are detected using semantic similarity between rolling message windows.
* Sentence embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.
* When similarity drops below a configurable threshold, a topic boundary is created.
* Each topic segment is summarized and stored as a topic checkpoint.

### 100 Message Checkpoints

* Every 100 chronological messages are grouped into a checkpoint.
* Each checkpoint is summarized independently.
* These checkpoints provide long-range memory retrieval.

### Retrieval System

* FAISS vector search is built over:

  * Topic summaries
  * Message checkpoint summaries
  * Raw message chunks
* User queries are embedded and searched across all indexes.
* Retrieved context is combined before generating answers.

### Persona Extraction

The system extracts:

* Habits
* Personal Facts
* Personality Traits
* Communication Style

Persona attributes are derived from explicit conversation evidence rather than assumptions.

### Chatbot

The chatbot:

* Uses topic checkpoint retrieval
* Uses message checkpoint retrieval
* Uses raw message retrieval
* Uses persona information
* Produces grounded answers with supporting evidence

---

## Topic Detection Methodology

Topic detection is performed independently for each conversation.

1. Messages are grouped by `conversation_id`.
2. A rolling window of recent messages is created.
3. Each window is embedded using `all-MiniLM-L6-v2`.
4. Cosine similarity is calculated between adjacent windows.
5. If similarity falls below a configurable threshold (default: 0.5), a new topic segment is created.
6. The completed segment becomes a topic checkpoint containing:

   * Topic ID
   * Start Message
   * End Message
   * Topic Summary

This ensures topic segmentation respects chronological order and conversation boundaries.

---

## Summarization

Each topic segment and message checkpoint is summarized using a configurable summarization backend.

Supported options:

* Local summarization models (`facebook/bart-large-cnn`)
* Extractive summarization
* Optional external API-based summarization

The system does not require external APIs to function.

---

## Example Queries

The chatbot can answer questions such as:

* What kind of person is this user?
* What are their habits?
* How do they communicate?
* What did the user say about Portland?
* What did they discuss regarding classic cars?
* What are the user's interests?
* Summarize conversations related to education.
* What personality traits are evident from the conversations?

---

## Technologies Used

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| Programming Language | Python 3.12                              |
| Data Processing      | Pandas                                   |
| Embeddings           | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search        | FAISS                                    |
| Summarization        | BART / Extractive Summaries              |
| Persona Extraction   | Rule-Based NLP                           |
| Frontend             | Streamlit                                |
| Storage              | JSON                                     |
| Retrieval            | Retrieval-Augmented Generation (RAG)     |

---

## Assumptions

### User 1 Persona

The dataset uses generic speaker labels (`User 1`, `User 2`) across independent conversations.

Persona extraction treats **User 1 as the primary speaker within each conversation** and aggregates recurring patterns across conversations.

### Chronological Processing

All messages are processed in chronological order using their message sequence.

Topic detection is performed within conversation boundaries, while 100-message checkpoints are generated globally across the message stream.

### External APIs

The system is designed to operate without requiring external LLM APIs.

Optional API integrations can be used for improved summarization quality, but the core RAG pipeline remains functional using local models and extractive techniques.

---

## Future Improvements

* More advanced topic segmentation models
* Hybrid dense + sparse retrieval
* Multi-user persona tracking
* Incremental memory updates
* Improved conversation summarization
* Real-time streaming chatbot responses
