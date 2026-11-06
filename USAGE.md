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

**Top bar**
- **User** box: whose memories and chats to use. Keep `demo`, or use your own name.
- **Memory** switch: turn off to see how it answers without long-term memory (no recall, no saving, no memory tools).

**Chats (left)**
- **+ New chat** starts a fresh conversation. Long-term memories still carry over.
- Click a chat to reopen it with its full history. Hover and click **×** to delete it (its memories stay).
- On a narrow window, open the list with the **Chats** button at the top left.

**Conversation (middle)**
- Replies stream in word by word.
- Under each reply: time to the first word, total time, how many memories were used (click it to see them), and a chip for each memory tool it used, e.g. "Saved: The user is vegetarian."
- You can ask it directly: "Remember that my sister is called Asha", "What do you know about me?", "Forget my sister's name".

**Side panel (right)**
- **Recalled**: memories used for the last reply, with scores.
- **Memories**: everything stored. Search, filter by category, **+ Add** one by hand, edit (✎) text, category or importance, delete (×), or **Export** them all as a JSON file.
- **Map**: memories plotted by meaning; hover a point to read it.
- **Logs**: turns, median reply time, median time to first word, memory tool uses, and every turn.

**What's saved where**
- Memories: `data/chroma`
- Chats: `data/threads.sqlite`
- Logs and chat history: `data/interactions.sqlite`

All three survive restarts. Delete the `data` folder to start completely fresh.

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

On Gemini's free tier, use **Gemini 3.5 Flash Lite**. It allows 15 requests a minute and 500 a day, and replies in about a second. Larger models like Gemini 3.8 Flash allow only 5 a minute and 20 a day.

Each chat message uses about **2 requests** (one to reply, one to save memories in the background), plus **1 more for each memory tool** it uses ("remember…", "forget…"). On a tight limit, set `AGENT_TOOLS=false` to save requests.

In `.env`:

```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite
LLM_THINKING_LEVEL=minimal
GOOGLE_API_KEY=your-key
```

You can check your limits and usage in Google AI Studio under **Dashboard → Rate Limit**. Daily limits reset at midnight US Pacific time (8am UK).

## Try it out

1. Tell it about yourself, e.g. "Hi, I'm Manju, I'm doing an MSc in Data Science at Queen Mary."
2. Wait a few seconds, then open **Memories**: you should see facts like "The user's name is Manju".
3. Click **+ New chat** and ask "What am I studying?". Open **Recalled** to see which memories it used.
4. Say "Remember that I'm vegetarian". A chip under the reply shows what it saved.
5. Say "Forget that I'm vegetarian". It searches, then deletes that memory.
6. Edit a memory in the **Memories** tab and ask about it again.
7. Restart the server and reopen your first chat from the sidebar: it's all still there.
8. Turn **Memory** off and ask again to compare.

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
| `AGENT_TOOLS` | `true` lets the assistant search, save and forget memories itself; `false` turns the tools off |
| `MAX_TOOL_ROUNDS` | Most rounds of tool calls per message before it must answer (default 3) |
| `CHECKPOINT_PATH` | Where chats are saved (default `data/threads.sqlite`); `:memory:` forgets them on restart |
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
| `429` / quota error | Free-tier limit reached. Wait a few minutes; use `--delay 5` for the evaluation. |
| `API key not valid` | Copy the key again from aistudio.google.com/apikey into `.env`, save, restart. |
| "temperature will be ignored" warnings | Harmless: Gemini 3 thinking models choose their own temperature. |
| `Address already in use` | A server is already running. Stop it with Control + C, or use `uvicorn app.api:app --port 8001`. |
| `No module named ...` | Run `pip install -r requirements.txt` with the environment active. |
| Chats disappear after a restart | Run `pip install -r requirements.txt` (it adds `langgraph-checkpoint-sqlite`), and check `CHECKPOINT_PATH` isn't `:memory:`. |
| Replies appear all at once instead of streaming | Some models or networks buffer output; the reply still arrives. Try another model, or check nothing between you and the server buffers responses. |
| It says it saved something but **Memories** doesn't show it | Click the **Memories** tab again to refresh, or check the **User** box matches. |

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
