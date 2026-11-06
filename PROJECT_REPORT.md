# Project report: Personalised AI Assistant with Contextual Recall

**Author:** Manju Swaroop V
**Stack:** Python, LangChain, LangGraph, Chroma DB, FastAPI, Hugging Face sentence-transformers, Gemini / OpenAI / Ollama
**Repository:** https://github.com/manjuv03/Personalized-AI-Assistant-with-Contextual-Recall

---

## 1. Summary

This is a chat assistant that remembers the people it talks to. If you tell it your name, your course or that you're vegetarian, it stores those facts and uses them in later conversations, even brand-new ones. It also keeps each conversation short by summarising older messages, so it stays fast as a chat grows.

It has two kinds of memory:

- **Long-term memory:** durable facts about each user, stored as vectors in Chroma DB and found again by meaning, not exact words.
- **Short-term (contextual) memory:** the current conversation, kept by LangGraph, with older turns folded into a running summary.

A FastAPI backend serves the chat and a web page that shows which memories were used for every reply, everything the assistant knows about you, a 2-D map of those memories, and a log of every turn with its response time. An evaluation script measures how well it remembers across conversations, with memory switched on versus off.

---

## 2. Why I built it

Most chatbots have one of two problems:

1. **They forget.** Close the chat and everything you said is gone. Next time you have to explain yourself again.
2. **They remember everything, badly.** Some pass the whole history to the model on every turn. That gets slow and expensive, eventually hits the model's context limit, and buries the important facts among small talk.

Real assistants should behave more like a good colleague: remember what matters about you, bring it up when it's relevant, and not recite your life story every time you ask a question.

I wanted to understand how to build that properly: how to decide what's worth remembering, how to find the right memory at the right moment, how to keep conversations small, and how to *measure* whether memory actually helps rather than just claiming it does.

---

## 3. Goals

| Goal | How the project meets it |
|---|---|
| Remember facts across separate conversations | Long-term memory in Chroma DB, filtered by user |
| Only bring back what's relevant | Semantic search plus re-ranking by importance and recency |
| Handle corrections ("actually, it's called Ember & Oak now") | Near-duplicate facts replace the old one |
| Keep replies fast as chats grow | Rolling summary of older turns; memory saving runs in the background |
| Make memory visible and editable | UI panels for recalled memories, all memories (with delete), memory map and logs |
| Measure the effect of memory | Evaluation comparing continuity with memory on vs off, plus latency |
| Work with free or local models | Pluggable providers: Gemini (free key), OpenAI, Ollama (fully local) |

---

## 4. Technology choices

| Tool | Used for | Why |
|---|---|---|
| **LangGraph** | The agent's workflow (recall → respond → summarise) | Explicit steps and state, a built-in checkpointer for per-conversation memory, and easy conditional branches |
| **LangChain** | Message types and model integrations | One interface for Gemini, OpenAI and Ollama |
| **Chroma DB** | Long-term memory store | Lightweight, runs locally, persistent, supports metadata filters (per user) and cosine search |
| **sentence-transformers (all-MiniLM-L6-v2)** | Turning text into vectors | Free, runs locally, fast on a laptop, good quality for short sentences |
| **FastAPI** | Web API and serving the UI | Fast, typed request models, background tasks for memory saving, automatic docs at `/docs` |
| **SQLite** | Interaction log | Zero setup, enough for one machine |
| **NumPy** | PCA for the memory map | Simple 2-D projection of embeddings |
| **pytest + GitHub Actions** | Tests on every push | Tests run offline with fakes, so CI needs no API key |

---

## 5. System architecture

```mermaid
flowchart TB
    subgraph Browser["Web UI (app/static/index.html)"]
        Chat["Chat window"]
        Panels["Side panel: Recalled · Memories · Map · Logs"]
    end

    subgraph API["FastAPI (app/api.py)"]
        ChatEP["POST /chat"]
        MemEP["/users/{id}/memories, /memory-map"]
        LogEP["/users/{id}/logs"]
        BG["Background task: save new facts"]
    end

    subgraph Core["Assistant (app/service.py)"]
        Graph["LangGraph agent (app/graph.py)"]
        Extract["Fact extractor (memory/extractor.py)"]
    end

    subgraph Stores["Storage"]
        Chroma[("Chroma DB<br/>long-term memory")]
        Ckpt[("LangGraph checkpointer<br/>per-thread messages + summary")]
        SQL[("SQLite<br/>interaction log")]
    end

    LLM["Chat model<br/>Gemini / OpenAI / Ollama<br/>(wrapped with retries)"]
    Emb["Embedding model<br/>all-MiniLM-L6-v2"]

    Chat --> ChatEP
    Panels --> MemEP
    Panels --> LogEP
    ChatEP --> Graph
    ChatEP --> BG --> Extract
    Graph --> Chroma
    Graph --> Ckpt
    Graph --> LLM
    Extract --> LLM
    Extract --> Chroma
    Chroma --- Emb
    ChatEP --> SQL
    MemEP --> Chroma
    LogEP --> SQL
```

### Code layout

```
app/
  api.py              FastAPI routes, error handling, start-up warm-up
  service.py          Assistant: runs the graph, saves memories, logs turns
  graph.py            LangGraph agent: recall -> respond -> summarise
  memory/
    long_term.py      Chroma DB store, ranking, de-duplication, PCA map
    short_term.py     Rolling-summary helpers
    extractor.py      Fact extraction prompt and JSON parsing
  retry.py            Retries for busy / rate-limited model calls
  llm.py              Model and embedding factories, reply-text helper
  logs.py             SQLite interaction log
  config.py           Settings from .env
  static/index.html   Web UI
eval/                 Evaluation scenarios, metrics and runner
tests/                Offline unit and end-to-end tests
```

---

## 6. How one message is handled

### 6.1 The flow

```mermaid
flowchart TD
    A([User sends a message]) --> B["POST /chat<br/>user_id, thread_id, message"]
    B --> C["LangGraph: load this thread's<br/>messages and summary"]
    C --> D{"Memory switched on?"}
    D -- yes --> E["recall: embed the message,<br/>search Chroma for this user"]
    E --> F["Re-rank hits by<br/>similarity, importance, recency;<br/>keep top 5 above threshold"]
    D -- no --> G
    F --> G["respond: build prompt =<br/>system rules + remembered facts<br/>+ summary + recent messages"]
    G --> H["Call chat model<br/>(retries if busy)"]
    H --> I{"More than 8 messages<br/>in this thread?"}
    I -- yes --> J["summarise: fold older messages<br/>into the running summary,<br/>remove them from state"]
    I -- no --> K
    J --> K["Return reply, recalled memories<br/>and latency to the browser"]
    K --> L["Log the turn in SQLite"]
    K --> M["Background: extract facts<br/>from the user's message"]
    M --> N{"Each fact: near-duplicate<br/>of an existing memory?"}
    N -- "yes, cosine ≥ 0.92" --> O["Update the old memory<br/>with the new wording"]
    N -- no --> P["Add a new memory"]
    O --> Q[("Chroma DB")]
    P --> Q
```

### 6.2 The same flow over time

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant G as LangGraph
    participant C as Chroma DB
    participant M as Chat model

    U->>API: "Hi, I'm Manju, doing an MSc at Queen Mary" (thread main)
    API->>G: invoke
    G->>C: recall (nothing yet)
    G->>M: prompt
    M-->>G: reply
    API-->>U: reply
    API->>M: (background) extract facts
    M-->>API: ["The user's name is Manju", "The user is doing an MSc ..."]
    API->>C: store facts

    U->>API: "What am I studying?" (thread new)
    API->>G: invoke (fresh thread, no history)
    G->>C: recall "What am I studying?"
    C-->>G: "The user is doing an MSc in Data Science at Queen Mary"
    G->>M: prompt + remembered fact
    M-->>G: "You're doing an MSc in Data Science at Queen Mary"
    API-->>U: reply + the memory it used
```

---

## 7. The memory logic in detail

### 7.1 Deciding what to remember

After each reply, the user's message goes to the chat model with an extraction prompt. The prompt asks for durable facts only (identity, preferences, goals, projects, corrections), one short third-person sentence each, with a category and an importance from 0 to 1, returned as JSON. Small talk and questions are ignored.

Models don't always return clean JSON, so the parser:

- strips code fences,
- takes the text between the first `[` and the last `]`,
- accepts plain strings as facts,
- clamps importance to 0–1 and maps unknown categories to `other`,
- returns an empty list rather than crashing on bad output.

This runs as a FastAPI background task, so the user never waits for it.

### 7.2 Storing without duplicates

Each fact is embedded and compared with the user's closest existing memory:

- **Cosine similarity ≥ 0.92:** it's the same fact, so the old record is updated with the new wording, the higher importance and a fresh timestamp. This is how corrections work: "my coffee shop is called Ember & Oak now" replaces "my coffee shop is called Ember".
- **Otherwise:** it's stored as a new memory.

Every memory carries `user_id` metadata, and every search filters on it, so users never see each other's facts.

### 7.3 Finding the right memory

Chroma returns the nearest facts by cosine similarity. I fetch three times as many as needed, then re-rank:

```
score = 0.75 × similarity + 0.15 × importance + 0.10 × recency
recency = 0.5 ^ (age in days / 30)        (halves every 30 days)
```

Facts scoring below 0.25 are dropped, and the top 5 go into the prompt.

Similarity carries most of the weight because relevance matters most. Importance and recency break ties. For example:

| Memory | Similarity | Importance | Age | Score |
|---|---|---|---|---|
| "The user's name is Priya" | 0.62 | 0.9 | today | 0.75×0.62 + 0.15×0.9 + 0.10×1.0 = **0.70** |
| "The user mentioned the weather" | 0.65 | 0.3 | 60 days | 0.75×0.65 + 0.15×0.3 + 0.10×0.25 = **0.56** |

The weather remark is slightly closer in meaning, but the name wins because it matters more and is more recent.

### 7.4 Keeping conversations small

```mermaid
flowchart LR
    A["Thread has 10 messages"] --> B{"More than<br/>KEEP_RECENT_MESSAGES (8)?"}
    B -- no --> C["Keep all"]
    B -- yes --> D["Oldest 2 messages"]
    B -- yes --> E["Newest 8 messages<br/>kept word for word"]
    D --> F["Model updates the<br/>running summary"]
    F --> G["Old messages removed<br/>from the thread"]
```

The prompt then holds the summary plus the last 8 messages, whatever the length of the conversation. Long-term facts don't depend on this, because they live in Chroma.

---

## 8. Reliability and speed

### 8.1 Busy or rate-limited models

While testing on Gemini's free tier, I hit "503: this model is currently experiencing high demand". Every model call now goes through a retry wrapper:

```mermaid
flowchart TD
    A["Call the model"] --> B{"Error?"}
    B -- no --> Z([Return reply])
    B -- yes --> C{"Busy or rate-limited?<br/>503, 429, quota, timeout"}
    C -- "no, e.g. bad key" --> X([Fail straight away<br/>with the real error])
    C -- yes --> D{"Retries left?<br/>(default 4)"}
    D -- yes --> E["Wait 1.5 s, 3 s, 6 s, 12 s...<br/>with a little random jitter"] --> A
    D -- no --> Y(["Return 503:<br/>'The model is busy right now'"])
```

The UI shows that message rather than a generic server error. The evaluation skips and counts busy probes instead of crashing.

### 8.2 Response time

| Technique | Effect |
|---|---|
| Fact extraction in a background task | The user waits for one model call, not two |
| Rolling summary | The prompt stays the same size however long the chat gets |
| Low Gemini 3 thinking level | Answers start much sooner |
| Reply length cap (1024 tokens) | Stops very long, slow answers |
| Warm-up at start-up | The embedding model and Chroma load before the first message |
| Local embeddings | Recall needs no network call |

The recall step (embedding + Chroma search) runs locally and is reported separately in the evaluation, because total response time is dominated by the hosted model and the network.

---

## 9. API and interface

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Send a message; returns the reply, memories used and latency |
| GET | `/users/{user_id}/memories` | List stored memories |
| POST | `/users/{user_id}/memories` | Add a memory by hand |
| DELETE | `/users/{user_id}/memories/{id}` | Forget a memory |
| GET | `/users/{user_id}/memory-map` | 2-D PCA coordinates for the map |
| GET | `/users/{user_id}/threads/{thread_id}/summary` | A thread's running summary |
| GET | `/users/{user_id}/logs` | Recent turns and average latency |

The web page has a chat window and four panels:

- **Recalled:** the memories used for the last reply, with similarity and final score.
- **Memories:** everything stored for the user, with a delete button for each item.
- **Map:** memories as points, projected to 2-D with PCA and coloured by category; similar facts sit close together.
- **Logs:** every turn with its time, response time, memories recalled and memories stored.

A **memory** tick box turns recall off, so you can compare answers with and without it.

---

## 10. Testing

The tests run offline. They use an in-memory vector store, hashing embeddings and a scripted model, so they need no network and no API key. GitHub Actions runs them on every push.

| Area | What is checked |
|---|---|
| Long-term memory | Relevant fact is recalled; users are isolated; near-duplicates update instead of adding; delete only works for the owner; unknown categories fall back |
| Ranking | Recency halves after 30 days; importance breaks ties |
| Extraction parsing | Code fences, out-of-range importance, garbage and empty output, plain strings |
| Short-term memory | Summary split keeps the recent window; nothing is summarised until it's full |
| Agent end-to-end | A fact told in one thread is recalled in another; memory can be switched off; long threads get summarised |
| Retries | Busy errors are retried, real errors are not, and the app gives up cleanly |
| Other | Interaction log round trip; keyword matching; Gemini content blocks; `.env` inline comments |

---

## 11. Evaluation method

```mermaid
flowchart TD
    A["8 personas in eval/scenarios.json<br/>e.g. Priya: nurse, vegetarian, half marathon"] --> B["Thread 'teach': tell the assistant<br/>each persona's facts; save memories"]
    B --> C["For each probe question<br/>e.g. 'Suggest a quick dinner'"]
    C --> D["Thread 'probe' (new, no history)<br/>ask with memory ON"]
    C --> E["Thread 'baseline' (new, no history)<br/>ask with memory OFF"]
    D --> F{"Reply contains an<br/>expected keyword?"}
    E --> G{"Reply contains an<br/>expected keyword?"}
    F --> H["Continuity with memory"]
    G --> I["Continuity without memory"]
    D --> J["Latency: recall step,<br/>full turn"]
    H --> K["eval/results.md"]
    I --> K
    J --> K
```

- **Continuity** is the share of probe questions answered correctly using facts from an earlier conversation. Each probe runs in a brand-new thread, so the conversation history can't help; only long-term memory can.
- **Baseline** asks the same question with memory switched off, so the difference shows what memory adds.
- **Keyword matching** uses whole words ("go" doesn't match "good") and prefixes where needed ("allerg*" matches allergy and allergic).
- **Latency** is reported as mean, median and p95, both for the full turn and for the recall step on its own.

There are 8 personas and 20 probe questions. Each can be run several times with `--runs`.

Run it with:

```bash
python -m eval.run_eval --runs 3 --delay 4
```

---

## 12. Outcome

### What works

- The assistant runs locally on a MacBook Air with Gemini and local embeddings, and replies through the web UI.
- Facts are extracted in the background, stored in Chroma DB and kept between restarts.
- Conversations are summarised once they pass the recent-message window.
- Busy and rate-limit errors are retried and reported clearly.
- All 24 tests pass on Python 3.12, and they run on every push through GitHub Actions.

### Results

<!-- Fill this table from eval/results.md after running the evaluation. -->

| Metric | With memory | Without memory |
|---|---|---|
| Continuity (probes answered from earlier sessions) | *to be measured* | *to be measured* |
| Mean response time | *to be measured* | *to be measured* |
| p95 response time | *to be measured* | *to be measured* |
| Recall step only (mean) | *to be measured* | n/a |

Model: Gemini 3.8 Flash. Embeddings: all-MiniLM-L6-v2. Personas: 8. Probes: 20.

---

## 13. Challenges and what I learned

| Problem | What happened | Fix / lesson |
|---|---|---|
| Python version | On Python 3.15, numpy and LangGraph wouldn't install, so tests failed or were skipped | Rebuilt the environment on Python 3.12. Check that dependencies support your Python version before upgrading. |
| Model overload | Gemini returned "503 high demand" and one reply took about 88 seconds | Added retries with backoff and a clear busy message, plus speed settings. Hosted models need resilience built in. |
| Reply format | Gemini 3 returns content as a list of blocks with signatures, which showed up raw in the UI | A small helper keeps only the text parts. Don't assume every provider returns a plain string. |
| Messy model output | Fact extraction sometimes wraps JSON in code fences or adds text | A tolerant parser that never crashes on bad output |
| Measuring honestly | "Better continuity" means nothing without a baseline | The evaluation always runs the same questions with memory off as well |

---

## 14. Limitations and next steps

**Limitations**

- Conversation threads live in memory and reset when the server restarts (long-term memories don't).
- Keyword-based scoring is simple; it can't tell a correct answer from one that mentions the keyword for another reason.
- Fact extraction costs a second model call per message.
- Built for one machine: no authentication and a single SQLite file.

**Next steps**

- Let the agent call memory tools itself (search, save, forget) instead of always recalling.
- Stream replies over Server-Sent Events, so text appears as it's generated.
- Store threads in a persistent checkpointer (SQLite or Postgres).
- Use an LLM judge alongside keywords in the evaluation.
- Add memory decay and periodic merging of related facts.
