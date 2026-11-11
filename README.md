# Self-Improving Intent Agent

Customer-support **intent classification** that gets better in two ways:

1. **Batch training (OPRO-style)** — Rewrite the system prompt across several rounds using train-set accuracy and mistake summaries.
2. **Live feedback** — Confirm or correct predictions in chat; new rules go into **ChromaDB** and are retrieved at classify time.

The underlying LLM model stays fixed (set `OPENROUTER_MODEL` in `.env`, default **`openai/gpt-4o-mini`** via [OpenRouter](https://openrouter.ai/)). What changes is the **prompt** (from training) and **lesson memory** (from training mistakes and user feedback).

## Architecture

```mermaid
flowchart LR
  subgraph batch [Batch training]
    T[data/train.json] --> E[evaluate]
    E --> M[memory.add lessons]
    E --> O[optimizer new prompt]
    O --> R[results.json]
  end
  subgraph runtime [Inference]
    R --> C[classify]
    DB[(memory_db Chroma)] --> C
    C --> LLM[OpenRouter LLM]
  end
  subgraph ui [Streamlit + FastAPI]
    APP[app.py] --> API[api.py]
    API --> batch
    API --> runtime
  end
```

| Piece | Role |
|-------|------|
| `results.json` | Best prompt and training log (written after `/train` or `main.py`) |
| `memory_db/` | Persistent Chroma collection `lessons` — short rules + reinforcement scores |
| `get_prompt()` | Uses `best_prompt` from `results.json`, or a default if you have not trained yet |

## Intent labels

`order_status`, `refund`, `complaint`, `greeting`, `cancel_order`, `delivery_info`, `account_help`, `other`

## Requirements

- Python **≥ 3.14** (`pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) (recommended)
- `OPENROUTER_API_KEY` in `.env`

## Setup

```bash
cd intent-agent
uv sync
```

Copy the template and set your key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

`.gitignore` already excludes `.env`, `memory_db/`, and `results.json`.

## Lint & Git

```bash
uv sync --group dev
uv run ruff check .
```

Before pushing to GitHub, see **[CONTRIBUTING.md](CONTRIBUTING.md)** (what to commit, branch workflow, first upload).

## Quick start (UI)

Use two terminals from the project root.

**Terminal 1 — API**

```bash
uv run uvicorn api:app --reload
```

Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Terminal 2 — Streamlit**

```bash
uv run streamlit run app.py
```

### Streamlit tabs

| Tab | What it does |
|-----|----------------|
| **Talk to the agent** | Classify a message → confirm **Yes** / **No** → optional intent picker on wrong guesses. Calls `/classify` and `/feedback`. |
| **Training and memory** | **Run training** (`POST /train`), score chart, prompt history, lessons table. |

**Sidebar — Memory** lists all lessons from `GET /lessons` (same store as the train tab). It is refreshed **after** tab logic runs so it stays in sync right after training (no need to send a chat message first).

## How batch training works

1. **Evaluate** — Score the current prompt on `data/train.json`; collect mistakes.
2. **Remember** — LLM writes a short rule per top mistake; near-duplicates are skipped (`memory.add`, distance under 0.2).
3. **Reinforce** — Lessons used during a call get score +1 / −1 when `learn=True` (training) or via `/feedback` (chat).
4. **Prune** — `memory.cleanup()` removes lessons with score ≤ −2.
5. **Optimize prompt** — Meta-prompt with history + mistakes → new text inside `<PROMPT></PROMPT>`.
6. **Pick best** — Highest train score → evaluate on `data/test.json` → write `results.json`.

### CLI only

```bash
uv run python main.py
```

Change rounds in `run_training(rounds=5)` or use `POST /train?rounds=5`.

## How live feedback works

1. `POST /classify` — Retrieves top‑k similar lessons, runs the classifier.
2. `POST /feedback` with `text`, `predicted`, `correct`:
   - **Confirmed** (`correct == predicted`) — Bump scores on lessons that were used; store a simple confirmation rule.
   - **Corrected** — Penalize used lessons; `make_lesson()` writes a new rule and saves it.

You can use chat **without** running training first; classification falls back to the default prompt until `results.json` exists.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/train?rounds=5` | Run full training loop; returns log, `best_prompt`, `test_score` |
| `GET` | `/results` | Read `results.json` (**404** until training has run) |
| `POST` | `/classify` | `{"text": "..."}` → intent (best or default prompt + memory) |
| `POST` | `/feedback` | `{"text", "predicted", "correct"}` → learning + optional `lesson` text |
| `GET` | `/lessons` | `[{"lesson", "score"}, ...]` |

### Example: classify + feedback

```bash
curl -s -X POST http://127.0.0.1:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"text": "Where is my parcel?"}'

curl -s -X POST http://127.0.0.1:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"text": "Where is my parcel?", "predicted": "order_status", "correct": "order_status"}'
```

## Data format

JSON arrays of `{ "text", "intent" }`:

- `data/train.json` — optimization + lesson creation during training  
- `data/test.json` — held-out metric after training  

## Project layout

| File | Role |
|------|------|
| `main.py` | Training loop |
| `evaluate.py` | Accuracy + mistakes; optional memory score updates |
| `optimizer.py` | New prompts + `make_lesson()` from errors |
| `agent.py` | Classify with labels + retrieved lessons |
| `config.py` | Typed settings from `.env` (Pydantic Settings) |
| `llm.py` | OpenRouter client |
| `memory.py` | Chroma persistence, search, dedupe, cleanup |
| `api.py` | HTTP API (train, classify, feedback, lessons) |
| `app.py` | Streamlit UI |
| `results.json` | Generated — `best_prompt`, `log`, `test_score` |
| `memory_db/` | Generated — Chroma data directory |

## Configuration

| Knob | Where |
|------|--------|
| Model / OpenRouter | `.env` — see `config.py` (`OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`, `OPENROUTER_API_KEY`) |
| Training rounds | `main.py` or `POST /train?rounds=` |
| Mistakes → lessons per round | `main.py` — top **3** |
| Mistakes in meta-prompt | `optimizer.py` — up to **5** |
| Lessons retrieved per message | `agent.py` — `memory.search(..., k=3)` |
| Dedupe threshold | `memory.py` — distance under **0.2** skips add |
| Prune threshold | `memory.py` — `cleanup(min_score=-2)` |

## Resetting ChromaDB (lesson memory)

Lessons are stored under **`memory_db/`** (collection **`lessons`**).

1. **Stop** uvicorn, Streamlit, and any `python main.py` run (avoid file locks).
2. Delete Chroma data:

   ```bash
   rm -rf memory_db
   ```

3. **(Optional)** Drop saved prompt and training log:

   ```bash
   rm -f results.json
   ```

4. Train or chat again — `memory_db/` is recreated on the next `memory.add()`.

`memory.cleanup()` only removes weak lessons (score ≤ −2), not the whole DB. There is no reset HTTP endpoint.

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Streamlit: “Could not reach the backend” | Start `uvicorn api:app` on port **8000** (matches `API` in `app.py`). |
| `/results` or classify errors about missing file | Run training once, or rely on default prompt until `results.json` exists. |
| Sidebar empty but train tab shows lessons | Update `app.py` — sidebar should load lessons **after** tabs; restart Streamlit after pulls. |
| Training is slow | Each round calls the LLM for every example plus meta-prompt / lesson writes; reduce `rounds` or shrink `train.json` for demos. |
| Duplicate or noisy lessons | Tighten dedupe in `memory.py` or run `rm -rf memory_db` for a clean memory slate. |

## Author

Chathuri Perera — chathuri@gluonforce.com
