# Using the assistant day to day (macOS)

Once setup is done (see the README), this is all you need each time.

## Start it

```bash
cd ~/Projects/Personalized-AI-Assistant-with-Contextual-Recall
source .venv/bin/activate
uvicorn app.api:app --reload
```

Wait for `Uvicorn running on http://127.0.0.1:8000`, then open **http://localhost:8000**.

## Use it

- **user** box: whose memory to use. Keep `demo`, or use your own name.
- **thread** box: the conversation. A new thread name starts a fresh chat; long-term memory still carries over.
- **memory** tick box: untick to see how it answers without long-term memory.
- Side panel:
  - **Recalled**: memories used for the last reply.
  - **Memories**: everything stored, with × to forget an item.
  - **Map**: memories plotted by meaning.
  - **Logs**: every turn with its response time.

Memories are saved in `data/chroma`, so they survive restarts. Conversation threads reset when the server restarts.

## Stop it

Press **Control + C** in the Terminal running the server.

## Get the latest code

```bash
cd ~/Projects/Personalized-AI-Assistant-with-Contextual-Recall
source .venv/bin/activate
git pull
pip install -r requirements.txt
python -m pytest
```

## Run the evaluation

```bash
python -m eval.run_eval --runs 1 --delay 4
```

Results go to `eval/results.md`. `--delay 4` waits 4 seconds between model calls, which keeps the free Gemini tier from rate-limiting.

## Settings (`.env`)

| Setting | What it does |
|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | Which chat model to use, e.g. `gemini` + `gemini-3.8-flash` |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` | Your key for that provider. Never commit or share it. |
| `LLM_MAX_RETRIES` | Retries when the model is busy (default 4) |
| `EMBEDDING_PROVIDER` | `huggingface` (local, free) or `gemini` / `openai` (hosted, smaller install) |
| `KEEP_RECENT_MESSAGES` | Messages kept word for word before older ones are summarised |

Restart the server after editing `.env`.

## Fixing common problems

| You see | What to do |
|---|---|
| `command not found: uvicorn` or `pytest` | Activate the environment: `source .venv/bin/activate` (look for `(.venv)`). |
| "The model is busy right now" | Google is overloaded. Wait a minute and resend, or switch `LLM_MODEL` to a Flash-Lite model. |
| `429` / quota error | Free-tier limit reached. Wait a few minutes; use `--delay 4` for the evaluation. |
| `API key not valid` | Copy the key again from aistudio.google.com/apikey into `.env`, save, restart. |
| `Address already in use` | A server is already running. Stop it with Control + C, or use `uvicorn app.api:app --port 8001`. |
| `No module named ...` | Run `pip install -r requirements.txt` with the environment active. |

## Free up disk space

The environment (`.venv`, mostly PyTorch) is the largest part at 1–2 GB, plus about 90 MB for the embedding model in `~/.cache/huggingface`. To remove them:

```bash
rm -rf ~/Projects/Personalized-AI-Assistant-with-Contextual-Recall/.venv
rm -rf ~/.cache/huggingface
```

To use the assistant again afterwards, recreate the environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
