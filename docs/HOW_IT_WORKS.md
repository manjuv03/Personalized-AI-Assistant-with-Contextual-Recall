# How it works: a guide for building this project yourself

This guide is for anyone who wants to understand this project well enough to build something like it from scratch. It explains the ideas first, then each technology and why it was chosen, then walks through the code file by file in the order a message travels through it. At the end there's a suggested build order and some exercises.

You should be comfortable with basic Python (functions, classes, dictionaries) and have used a terminal. No prior experience with LLMs, vector databases or LangGraph is assumed.

**Contents**

1. [The problem in one picture](#1-the-problem-in-one-picture)
2. [Core ideas you need](#2-core-ideas-you-need)
3. [The technologies and why each one is used](#3-the-technologies-and-why-each-one-is-used)
4. [The big picture](#4-the-big-picture)
5. [Following one message through the code](#5-following-one-message-through-the-code)
6. [File-by-file walkthrough](#6-file-by-file-walkthrough)
7. [Design decisions and the alternatives](#7-design-decisions-and-the-alternatives)
8. [Testing without an API key](#8-testing-without-an-api-key)
9. [Measuring whether memory helps](#9-measuring-whether-memory-helps)
10. [Build it yourself: a suggested order](#10-build-it-yourself-a-suggested-order)
11. [Exercises](#11-exercises)
12. [Glossary](#12-glossary)

---

## 1. The problem in one picture

A large language model (LLM) like Gemini or GPT has **no memory of its own**. Every request is independent. If you want it to "remember" something, you have to send that information again in the prompt.

So every chatbot has to decide *what* to send. There are two naive answers, and both are bad:

| Approach | What happens |
|---|---|
| Send nothing from before | The bot forgets your name the moment you start a new chat |
| Send the entire history every time | Slow, expensive, eventually too long for the model, and the important facts get buried in small talk |

This project does something in between:

- It keeps **long-term memory**: short facts about you ("The user is doing an MSc in Data Science at Queen Mary"), stored in a database and searched by meaning.
- It keeps **short-term memory**: the current conversation, with older parts shrunk into a summary.
- On each message, it sends the model only the **relevant** facts, the summary and the last few messages.

---

## 2. Core ideas you need

### 2.1 Prompts and messages

When you call a chat model you send a list of messages, each with a role:

- **system**: instructions for how the model should behave ("You are a friendly assistant...")
- **human** (or user): what the person typed
- **ai** (or assistant): what the model said before

The model reads the whole list and writes the next AI message. Everything this project does is about building the right list.

### 2.2 Embeddings: turning text into numbers that capture meaning

An **embedding model** turns a piece of text into a list of numbers (a vector). This project uses `all-MiniLM-L6-v2`, which produces 384 numbers per sentence.

The useful property: **sentences with similar meaning get similar vectors**, even when they share no words.

- "What am I studying?" and "The user is doing an MSc in Data Science" have different words but are about the same thing, so their vectors point in a similar direction.
- "What am I studying?" and "The user has a cat called Biscuit" point in different directions.

### 2.3 Cosine similarity: how close two meanings are

To compare two vectors we use **cosine similarity**: the cosine of the angle between them.

```
cosine(a, b) = (a · b) / (|a| × |b|)
```

- 1.0 means the same direction (same meaning)
- 0 means unrelated
- negative means opposite

For short sentences, real scores are often lower than you'd expect. In this app, "What am I studying?" against "The user is pursuing an MSc in Data Science at Queen Mary" scored about **0.29**, which was still the best match, so it was recalled. That's why the code ranks matches rather than demanding a high score.

### 2.4 Vector databases

A **vector database** stores text together with its embedding, and answers the question "which stored items are closest to this vector?" quickly. This project uses **Chroma DB**. Each stored memory also has **metadata** (user id, category, importance, timestamps), and searches can filter on metadata, so one user never sees another's memories.

### 2.5 Retrieval before generation (the RAG pattern)

"Retrieval-augmented generation" means: before asking the model, **retrieve** relevant information and put it in the prompt. Usually RAG retrieves from documents; here it retrieves from facts about the user. The steps are the same:

1. embed the question
2. search the vector database
3. add the best results to the prompt
4. generate the answer

### 2.6 Agents and graphs

An **agent** here just means "a program that runs a few steps around an LLM call". LangGraph lets you write those steps as a **graph**: nodes are functions, edges say what runs next, and a shared **state** dictionary flows through them. A **checkpointer** saves the state after each run, keyed by a **thread id**, which is how a conversation continues where it left off.

---

## 3. The technologies and why each one is used

| Technology | What it is | Why this project uses it | What you could use instead |
|---|---|---|---|
| **Python 3.12** | The language | The LLM and data ecosystem lives here. 3.10–3.13 are supported; newer versions may lack builds of numpy/Chroma. | — |
| **LangChain (core)** | Standard message types and model interfaces | One `invoke()` interface for Gemini, OpenAI and Ollama, so switching models is a config change | Call each provider's SDK directly |
| **LangGraph** | A library for building LLM workflows as graphs with state | Clear steps (recall → respond → summarise), conditional branches, and a checkpointer that stores each conversation | A hand-written loop plus your own conversation store |
| **Chroma DB** | A local, persistent vector database | Runs on a laptop with no server, saves to disk, supports cosine search and metadata filters | FAISS (no metadata filters built in), pgvector (needs Postgres), Qdrant, Pinecone (hosted) |
| **sentence-transformers / all-MiniLM-L6-v2** | A small local embedding model | Free, about 90 MB, fast on a CPU, good for short sentences, no API calls (so no rate limits for recall) | OpenAI or Gemini embedding APIs (smaller install, but network calls and quotas) |
| **Gemini / OpenAI / Ollama** | The chat models | Gemini has a free tier; OpenAI is cheap and reliable; Ollama runs fully locally | Any LangChain-supported model |
| **FastAPI** | A Python web framework | Typed request models, automatic API docs at `/docs`, and **background tasks**, used so saving memories doesn't slow replies | Flask (no built-in typing/validation), Django (heavier) |
| **Uvicorn** | The server that runs FastAPI | Standard choice, auto-reload during development | Hypercorn, Gunicorn with workers |
| **SQLite** | A file-based SQL database | Interaction logs with zero setup | Postgres for multi-user deployments |
| **NumPy** | Numerical arrays | PCA (via SVD) to draw the 2-D memory map | scikit-learn's PCA |
| **Plain HTML/CSS/JS** | The web UI | One file, no build step, easy to read | React, Streamlit, Gradio |
| **pytest + GitHub Actions** | Testing and CI | Tests run offline with fakes, so CI needs no API key | unittest |

---

## 4. The big picture

```mermaid
flowchart LR
    UI[Browser] --> API[FastAPI]
    API --> G[LangGraph agent]
    G <--> C[(Chroma DB)]
    G <--> K[(Thread state)]
    G --> M[Chat model]
    API -.-> E[Fact extractor] --> C
    E --> M
    API --> L[(SQLite logs)]
    classDef default fill:#dbe6f0,stroke:#1f5b8f,color:#15212b
    classDef io fill:#1f5b8f,stroke:#1f5b8f,color:#ffffff
    classDef store fill:#a8661a,stroke:#a8661a,color:#ffffff
    class C,K,L store
```

Two flows happen for every message:

1. **The reply flow (solid lines)**: the user waits for this. Recall facts, build the prompt, call the model, return the answer.
2. **The memory flow (dashed line)**: runs in the background after the reply is sent. Ask the model which facts in the message are worth keeping, then store them.

---

## 5. Following one message through the code

Say user `demo` types **"What am I studying?"** in thread `test2`.

```mermaid
%%{init: {'theme':'base','themeVariables':{'actorBkg':'#dbe6f0','actorBorder':'#1f5b8f','actorTextColor':'#15212b','signalColor':'#56636e','signalTextColor':'#56636e','noteBkgColor':'#f6ead9','noteBorderColor':'#a8661a','sequenceNumberColor':'#ffffff'}}}%%
sequenceDiagram
    participant B as Browser
    participant A as api.py
    participant S as service.py
    participant G as graph.py
    participant C as Chroma
    participant M as Model
    B->>A: POST /chat
    A->>S: assistant.chat()
    S->>G: graph.invoke()
    G->>C: recall
    C-->>G: top facts
    G->>M: prompt
    M-->>G: reply
    G-->>S: final state
    S-->>A: ChatResult
    A-->>B: JSON reply
    A->>S: remember_safely() in background
```

Step by step:

1. **Browser** (`app/static/index.html`) sends `POST /chat` with `{"user_id": "demo", "thread_id": "test2", "message": "What am I studying?", "use_memory": true}`.
2. **`app/api.py` → `chat()`** validates the body with a Pydantic model, then calls `assistant.chat(...)`.
3. **`app/service.py` → `Assistant.chat()`** builds a LangGraph config with `thread_id = "demo:test2"` and `user_id = "demo"`, starts a timer and calls `graph.invoke(...)` with the new human message.
4. **The checkpointer** loads any earlier messages and summary for `demo:test2`. LangGraph appends the new message to the state.
5. **`recall` node** takes the latest human message and calls `memory.recall("demo", "What am I studying?")`.
6. **`app/memory/long_term.py` → `LongTermMemory.recall()`** embeds the question, asks Chroma for the 15 nearest memories belonging to `demo`, re-ranks them with the scoring formula, drops weak ones and returns the top 5.
7. **`respond` node** builds the prompt: system rules, "What you remember about this user: - The user is pursuing an MSc...", the summary if there is one, then the conversation messages. It calls the chat model.
8. **`app/retry.py`** wraps that call: if the provider says it's busy, it waits and tries again.
9. **`needs_summary`** checks the thread length. If it has reached 16 messages, the `summarise` node folds the oldest ones into the summary and removes them.
10. **Back in `Assistant.chat()`**, the reply text is pulled out (`message_text`), latency is measured, and the turn is written to the SQLite log.
11. **`api.py`** returns the reply, the recalled memories and the latency as JSON, and schedules `remember_safely()` as a background task.
12. **Background:** `Assistant.remember()` sends the message to the model with the extraction prompt, parses the JSON list of facts, and calls `memory.add()` for each. Here "What am I studying?" contains no facts, so nothing is stored.
13. **Browser** shows the reply, the response time and how many memories were used; the Recalled panel lists them with their scores.

---

## 6. File-by-file walkthrough

```
app/
  config.py        settings
  llm.py           model + embedding factories, reply text helper
  retry.py         retries for busy models
  memory/
    long_term.py   Chroma store, ranking, duplicates, memory map
    extractor.py   deciding what to remember
    short_term.py  summary helpers
  graph.py         the LangGraph agent
  service.py       Assistant class tying it together
  logs.py          SQLite interaction log
  api.py           FastAPI routes
  static/index.html  web UI
eval/              evaluation
tests/             tests
```

### 6.1 `app/config.py`: settings

All tunable values live in one frozen dataclass, `Settings`, read from environment variables. A small loader reads `.env` first (so `python-dotenv` isn't required) and ignores comments at the end of lines.

Key settings and what they control:

| Setting | Default | Controls |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | `openai`, `gpt-4o-mini` | Which chat model |
| `LLM_MAX_RETRIES` | 4 | Retries when the model is busy |
| `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `LLM_THINKING_LEVEL` | 1024, 60, `low` | Speed |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `huggingface`, MiniLM | How text becomes vectors |
| `VECTOR_BACKEND` | `chroma` | `chroma` on disk, or `memory` for tests |
| `RECALL_TOP_K`, `RECALL_MIN_SCORE` | 5, 0.25 | How many facts go into the prompt, and the cut-off |
| `DUPLICATE_THRESHOLD` | 0.92 | When a new fact replaces an old one |
| `KEEP_RECENT_MESSAGES` | 8 | Short-term memory window |

**Why a dataclass?** Typed, one place to look, and easy to override in tests (`Settings(keep_recent_messages=4)`).

### 6.2 `app/llm.py`: models and embeddings

Three things live here:

- **`get_chat_model(settings)`** builds the chosen provider's LangChain chat model (imported lazily, so you only install the one you use) and wraps it in `RetryingChatModel`. For Gemini it also sets the thinking level and output cap.
- **`get_embeddings(settings)`** returns an object with `embed_query(text)` and `embed_documents(texts)`. Options: Hugging Face (local), OpenAI, Gemini, or `HashEmbeddings`.
- **`HashEmbeddings`** is a tiny offline embedder for tests: it hashes each word into one of 512 slots. It only captures word overlap, not meaning, but it's deterministic and needs no download.
- **`message_text(message)`** returns the plain text of a reply. Some providers (Gemini 3) return a list of content blocks, `[{"type": "text", "text": "..."}, ...]`, rather than a string; this keeps only the text parts.

### 6.3 `app/retry.py`: surviving busy models

Hosted models on free tiers often reply "503 high demand" or "429 quota", or drop the connection. `RetryingChatModel.invoke()`:

1. calls the real model
2. on error, checks `is_transient(error)` by looking for markers like `503`, `429`, `quota`, `timeout`, `disconnected` in the error text
3. if transient and retries remain, sleeps 1.5 s, 3 s, 6 s, 12 s (times a random 0.8–1.2 "jitter", so many clients don't retry in sync) and tries again
4. if not transient (bad key, wrong model name), raises immediately, since retrying won't help
5. after the last retry, raises `ModelBusyError`, which the API turns into a friendly 503

### 6.4 `app/memory/long_term.py`: the heart of the project

**Data shapes**

- `MemoryRecord`: one fact: `id`, `user_id`, `text`, `category` (profile / preference / goal / project / other), `importance` (0–1), `created_at`, `updated_at`, `times_recalled`.
- `RecalledMemory`: a record plus its `similarity` and final `score`.

**Backends** share one small interface (`upsert`, `query`, `list`, `delete`):

- `ChromaBackend`: one persistent collection with cosine distance. Every query filters `where={"user_id": ...}`. Chroma returns cosine *distance*, so similarity is `1 - distance`.
- `InMemoryBackend`: plain Python with the same interface, for tests.

This is the **strategy pattern**: `LongTermMemory` doesn't care which backend it has.

**Adding a fact: `LongTermMemory.add()`**

```
embedding = embed(text)
closest = backend.query(user_id, embedding, k=1)
if closest similarity >= 0.92:
    update that record (new wording, higher importance, new timestamp)
else:
    create a new record
backend.upsert(record, embedding)
```

The update branch is how corrections work: "My coffee shop is called Ember & Oak now" is very close to "My coffee shop is called Ember", so it replaces it instead of keeping both.

**Recalling: `LongTermMemory.recall()`**

```
hits = backend.query(user_id, embed(query), k = top_k * 3)    # over-fetch
for each hit:
    score = 0.75*similarity + 0.15*importance + 0.10*recency
keep score >= 0.25, sort by score, return top 5
```

with `recency = 0.5 ** (age_days / 30)`, so a memory loses half its recency weight every 30 days.

Why over-fetch? Chroma ranks only by similarity. A slightly less similar but much more important fact might be 7th by similarity and 2nd by score, so we fetch 15 and re-rank.

Why these weights? Relevance should dominate (0.75). Importance (0.15) lets "my name is Priya" beat a passing remark. Recency (0.10) gently favours newer facts without forgetting old ones. They're tunable; the exercises suggest experimenting.

**Memory map: `projection()`** stacks the user's embeddings into a matrix, centres it, and uses SVD to project onto the top 2 principal components. Points close together on the map are similar in meaning.

### 6.5 `app/memory/extractor.py`: deciding what to remember

`extract_facts(model, message)` sends the message with a prompt that asks for durable facts only, one third-person sentence each, as JSON:

```json
[{"text": "The user's name is Priya.", "category": "profile", "importance": 0.9}]
```

`parse_extracted_facts(raw)` is deliberately forgiving, because models don't always follow formats exactly. It strips code fences, takes the text between the first `[` and last `]`, accepts plain strings, clamps importance to 0–1, maps unknown categories to `other`, and returns `[]` instead of crashing.

**Why use the LLM for extraction instead of rules?** Rules can't cope with the variety of ways people say things. The cost is one extra model call per message, which is why it runs in the background.

### 6.6 `app/memory/short_term.py`: summaries

- `split_for_summary(messages, keep_recent)` returns `(older, recent)`.
- `build_summary_prompt(summary, older)` asks the model to fold the older messages into the existing summary, keeping names, decisions and open questions.

### 6.7 `app/graph.py`: the LangGraph agent

**State** is a typed dictionary:

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    recalled: list[dict]
```

`Annotated[..., add_messages]` is a **reducer**: when a node returns `{"messages": [new]}`, LangGraph *appends* rather than replaces. Returning `RemoveMessage(id=...)` deletes a message, which is how summarising trims the thread.

**Nodes** are plain functions that take the state (and config) and return a partial update:

- `recall`: reads `user_id` and `use_memory` from the config, searches long-term memory, returns `{"recalled": [...]}`.
- `respond`: builds the system message from the rules, recalled facts and summary; calls the model; returns `{"messages": [reply]}`.
- `summarise`: folds older messages into `summary` and removes them.

**Edges** wire them up:

```python
graph.add_edge(START, "recall")
graph.add_edge("recall", "respond")
graph.add_conditional_edges("respond", needs_summary, {"summarise": "summarise", END: END})
graph.add_edge("summarise", END)
```

`needs_summary` returns `"summarise"` once the thread holds `2 × KEEP_RECENT_MESSAGES` messages. Batching like this means the extra summary call happens about once every 4 turns rather than every turn.

**Checkpointer:** `graph.compile(checkpointer=InMemorySaver())` saves state per `thread_id`. It lives in memory, so threads reset on restart; long-term memories don't, because they're in Chroma.

### 6.8 `app/service.py`: the `Assistant` class

The one object the API talks to. It builds the model, memory, log and graph, and offers:

- `chat(user_id, thread_id, message, use_memory)`: runs the graph, times it, logs it and returns a `ChatResult`.
- `remember(user_id, message, log_id)`: extracts and stores facts.
- `remember_safely(...)`: the background version, which logs errors instead of raising them.
- `summary(user_id, thread_id)`: reads a thread's summary from the checkpointer.

Keeping this separate from `api.py` means the evaluation script and tests can use the assistant without a web server. This is **separation of concerns**.

### 6.9 `app/logs.py`: interaction log

A small SQLite table with one row per turn: time, user, thread, message, reply, recalled memories, stored memories and latency. A lock makes it safe when FastAPI runs handlers on several threads.

### 6.10 `app/api.py`: the web API

- `lifespan` warms up the embedding model when the server starts, so the first message isn't slow.
- `get_assistant()` is cached (`lru_cache`), so there's one `Assistant` per process.
- Exception handlers turn `ModelBusyError` into a 503 with a friendly message, and anything else into a 500 with a short description.
- `POST /chat` replies, then adds `remember_safely` as a **background task**: FastAPI sends the response first and runs the task afterwards.
- Other routes list, add and delete memories, return the memory map, a thread summary and logs.

### 6.11 `app/static/index.html`: the UI

One file with no framework. It calls the API with `fetch`, renders messages, and fills four panels (Recalled, Memories, Map, Logs). The map is drawn on a `<canvas>` from the projection coordinates. Colours come from CSS variables with a dark-mode variant.

---

## 7. Design decisions and the alternatives

| Decision | Chosen | Why | Trade-off |
|---|---|---|---|
| When to recall | Always, before replying | Simple and predictable | Wastes a search when no memory is relevant; an agent that decides itself is in the exercises |
| What to store | Extracted facts, not raw messages | Facts are short, clean and easy to de-duplicate | Needs a model call per message |
| When to store | Background task | User doesn't wait | A fact said in one message isn't available until a moment later |
| Ranking | Weighted similarity + importance + recency | Relevance first, with sensible tie-breakers | Weights are hand-picked |
| Duplicates | Replace when cosine ≥ 0.92 | Handles corrections and repeats | Two genuinely different but similar facts could merge |
| Short-term memory | Batched rolling summary | Prompt stays small; fewer extra calls | Fine details in old turns can be lost |
| Embeddings | Local MiniLM | Free, fast, no rate limits | 1–2 GB install because of PyTorch |
| One collection, filtered by user | Yes | Simple; Chroma filters are fast | For millions of users you'd shard |

---

## 8. Testing without an API key

Tests swap the expensive parts for fakes:

- `InMemoryBackend` instead of Chroma
- `HashEmbeddings` instead of a real embedding model
- a `ScriptedModel` that returns canned replies depending on the prompt (fact JSON for extraction prompts, a fixed summary for summary prompts, "Noted!" otherwise)

That makes it possible to test the whole agent end to end, e.g. "a fact told in thread t1 is recalled in thread t2", in under a second, with no network. GitHub Actions runs the tests on every push.

Run them with:

```bash
python -m pytest
```

---

## 9. Measuring whether memory helps

`eval/run_eval.py`:

1. For each persona in `eval/scenarios.json` (e.g. Priya: nurse, vegetarian, training for a half marathon), it tells the assistant the facts in a "teach" thread and saves memories.
2. For each probe question (e.g. "Suggest a quick dinner I could make tonight"), it asks in a **brand-new thread**, so only long-term memory can help, once with memory on and once with it off.
3. A probe passes if the reply contains an expected keyword (whole-word match; `allerg*` style prefixes allowed).
4. It reports continuity with vs without memory, and latency (mean, median, p95) for the full turn and the recall step on its own.

```bash
python -m eval.run_eval --runs 1 --delay 5
```

The baseline matters: without it, "85% continuity" could mean the model was guessing well, not remembering.

---

## 10. Build it yourself: a suggested order

Build in small steps, getting each working before the next:

1. **Call a model.** A script that sends one message to Gemini or OpenAI through LangChain and prints the reply.
2. **Embeddings and similarity.** Embed a few sentences with sentence-transformers and print the cosine similarity between pairs. Get a feel for the numbers.
3. **Store and search.** Put five facts in a Chroma collection with a `user_id` in metadata; query with a question; print the nearest. Then try two users.
4. **Ranking.** Add importance and timestamps; implement the scoring formula; check that ordering changes as expected.
5. **Prompt with memory.** Put the top facts into a system message and ask the model a question that needs them.
6. **Extraction.** Write the extraction prompt and a forgiving JSON parser; test it on ten varied messages.
7. **De-duplication.** Before storing, check the nearest existing fact and update it if it's very close.
8. **LangGraph.** Move recall and respond into nodes, add a checkpointer, and continue a conversation by thread id.
9. **Summarising.** Add the conditional summary node.
10. **API.** Wrap it in FastAPI with `POST /chat` and background memory saving.
11. **UI.** A simple HTML page that calls the API; then add the panels.
12. **Retries.** Wrap the model with retry logic once you've seen your first 503.
13. **Tests.** Fakes for the model, embeddings and store; then end-to-end tests.
14. **Evaluation.** Personas, probes, a baseline, and a results file.

---

## 11. Exercises

1. **Tune the ranking.** Change the weights in `long_term.py` and rerun the evaluation. Does continuity go up or down?
2. **Memory as tools.** Give the model `search_memory`, `save_memory` and `forget_memory` tools and let it decide when to use them, instead of always recalling.
3. **Persistent threads.** Replace `InMemorySaver` with LangGraph's SQLite checkpointer so conversations survive restarts.
4. **Streaming.** Add a `/chat/stream` endpoint using Server-Sent Events, so text appears as it's generated.
5. **Better scoring.** Use a second model call as a judge in the evaluation, alongside keyword matching.
6. **Forgetting.** Add decay: lower the importance of memories that are never recalled.
7. **Merge related facts.** Periodically combine facts like "likes tea" and "drinks green tea daily" into one.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **LLM** | Large language model; generates text from a prompt |
| **Prompt** | The list of messages sent to the model |
| **Token** | A chunk of text (roughly ¾ of a word) that models read and write; limits and costs are counted in tokens |
| **Embedding** | A list of numbers representing the meaning of a text |
| **Cosine similarity** | How closely two embeddings point the same way (1 = same, 0 = unrelated) |
| **Vector database** | A database that finds stored items nearest to a given embedding |
| **Metadata filter** | Restricting a vector search to items with certain metadata, e.g. one user |
| **RAG** | Retrieval-augmented generation: fetch relevant information, then generate |
| **Node / edge / state** | LangGraph's building blocks: a step, what runs next, and the shared data |
| **Reducer** | A rule for combining a node's output with existing state (e.g. append messages) |
| **Checkpointer** | Saves graph state per thread so a conversation can continue |
| **Thread** | One conversation, identified by a thread id |
| **Background task** | Work FastAPI runs after sending the response |
| **Exponential backoff** | Waiting longer after each failed retry (1.5 s, 3 s, 6 s...) |
| **PCA** | Principal component analysis; squeezes many dimensions into 2 for plotting |
| **p95 latency** | 95% of requests were faster than this |
