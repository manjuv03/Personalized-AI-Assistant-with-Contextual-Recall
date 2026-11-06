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

## Recommended model (free)

On Gemini's free tier, use **Gemini 3.5 Flash Lite**. It allows 15 requests a minute and 500 a day, and replies in about a second. Larger models like Gemini 3.8 Flash allow only 5 a minute and 20 a day, and each chat message uses about 2 requests (one to reply, one to save memories), so they run out after about 10 messages.

In `.env`:

```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
LLM_THINKING_LEVEL=minimal
GOOGLE_API_KEY=your-key
```

You can check your limits and usage in Google AI Studio under **Dashboard → Rate Limit**. Daily limits reset at midnight US Pacific time (8am UK).

## Try it out

1. Set **thread** to `main` and tell it about yourself, e.g. "Hi, I'm Manju, I'm doing an MSc in Data Science at Queen Mary."
2. Wait a few seconds, then open **Memories**: you should see facts like "The user's name is Manju".
3. Change **thread** to something new, e.g. `test2`, and ask "What am I studying?"
4. Open **Recalled** to see which memories it used and their scores.
5. Untick **memory** and ask again to compare.

## Run the evaluation

```bash
python -m eval.run_eval --runs 1 --delay 5
```

Run it in a second Terminal window while nothing else is using the model. It takes about 10–15 minutes and uses about 90 requests. Results go to `eval/results.md`. `--delay 5` waits 5 seconds between model calls, which keeps you under 15 requests a minute on the free tier. For a bigger test, `--runs 5` asks 100 questions (about 450 requests, close to the free daily limit).

## Settings (`.env`)

| Setting | What it does |
|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | Which chat model to use, e.g. `gemini` + `gemini-3.5-flash-lite` |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` | Your key for that provider. Never commit or share it, or show it in screenshots. |
| `LLM_THINKING_LEVEL` | `minimal` or `low` is fastest; `default` leaves it to the model |
| `LLM_MAX_RETRIES` | Retries when the model is busy (default 4) |
| `EMBEDDING_PROVIDER` | `huggingface` (local, free) or `gemini` / `openai` (hosted, smaller install) |
| `KEEP_RECENT_MESSAGES` | Messages kept word for word; older ones are summarised once the thread reaches twice this |

Restart the server after editing `.env`.

## Fixing common problems

| You see | What to do |
|---|---|
| `command not found: uvicorn` or `pytest` | Activate the environment: `source .venv/bin/activate` (look for `(.venv)`). |
| "The model is busy right now" | Google is overloaded or you hit a limit. Wait a minute and resend, or switch `LLM_MODEL` to a Flash-Lite model. |
| Replies take a long time | You may be near the rate limit, so the app is waiting and retrying. Check AI Studio → Rate Limit, and use Gemini 3.5 Flash Lite. |
| It doesn't remember something | Check **Memories**. If the fact isn't there, saving failed (often a rate limit): tell it again and wait a few seconds. |
| `untracked working tree files would be overwritten` on `git pull` | Your own `eval/results.md` clashes with the one in the repo. Run `mv eval/results.md eval/results-local.md`, then `git pull` again. |
| `429` / quota error | Free-tier limit reached. Wait a few minutes; use `--delay 4` for the evaluation. |
| `API key not valid` | Copy the key again from aistudio.google.com/apikey into `.env`, save, restart. |
| "temperature will be ignored" warnings | Harmless: Gemini 3 thinking models choose their own temperature. |
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
