# Self-Improving Intent Agent

A customer-message intent classifier for reconciliation workflows. It uses an
OpenRouter-hosted language model for classification and improves its behavior
through prompt optimization and a persistent collection of lessons. It does
not fine-tune the language model.

## What it does

- Classifies messages into a fixed set of intent labels.
- Trains an improved prompt from the examples in `data/train.json`.
- Stores and retrieves lessons from previous mistakes using ChromaDB.
- Accepts user feedback to reinforce or correct retrieved lessons.
- Provides a FastAPI backend and a Streamlit interface.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An [OpenRouter API key](https://openrouter.ai/keys)

## Setup

From the project directory:

```bash
uv sync --group dev
cp .env.example .env
```

Edit `.env` and add your API key:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
# Optional:
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEBUG=false
```

The model and base URL can be changed through these environment variables.
Settings are loaded from `.env`; restart the API or training process after
changing them. Do not commit `.env` or share your API key.

### Debug output

`DEBUG` is off by default. Set `DEBUG=true` in `.env` or in the process
environment to print training progress and memory lesson diagnostics:

```bash
DEBUG=true uv run python -m intent_agent.training.train
```

The `scripts/inspect_chroma.py` command always prints its report; printing the
database inspection is that script's purpose.

## Run the application

Start the API in one terminal:

```bash
uv run uvicorn intent_agent.api.app:app --reload
```

The API and interactive OpenAPI documentation are available at
[http://127.0.0.1:8000](http://127.0.0.1:8000) and
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Start the Streamlit interface in a second terminal:

```bash
uv run streamlit run src/intent_agent/ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). The UI requires the API to
be running. The API has no authentication configured, so keep it on a trusted,
local network.

## Train the classifier

Run five training rounds from the command line:

```bash
uv run python -m intent_agent.training.train
```

Training can also be started in the UI or with `POST /train?rounds=5`. Each
round:

1. Evaluates the current prompt against every example in `data/train.json`.
2. Updates scores for lessons used during those classifications.
3. Asks the language model to create lessons for up to the first three
   mistakes, then stores new, non-duplicate lessons.
4. Removes lessons with a score of -2 or lower.
5. Asks the language model to rewrite the prompt using the prompt history and
   up to five mistakes.

After the rounds, the best-scoring prompt is evaluated against
`data/test.json`. Training writes the prompt and scores to
`storage/results.json`. The test set is used for evaluation, not for learning.

## Intent labels

The classifier returns one of these labels:

| Label | Meaning |
|---|---|
| `match_invoice` | Match a payment or transaction to an invoice |
| `unmatched_items` | Find or explain unmatched items |
| `upload_document` | Upload a document |
| `amount_mismatch` | Investigate an amount discrepancy |
| `recon_status` | Ask about reconciliation status |
| `user_access` | Ask about user access or permissions |
| `audit_query` | Ask about an audit log or activity |
| `login_help` | Get help signing in |
| `other` | Any other intent |

The label list is defined in `src/intent_agent/agent/classify.py`.

## Feedback and memory

Classification can use the best prompt saved by training and retrieve up to
three relevant lessons from ChromaDB. Both options can be disabled for
`POST /classify` using its `use_best_prompt` and `use_memory` fields.

Feedback is submitted to `POST /feedback` with the message, predicted label,
and correct label:

- If the prediction was correct, each retrieved lesson's score increases by 1.
- If it was incorrect, each retrieved lesson's score decreases by 1, and the
  language model creates a lesson for the correction.
- New lessons start with a score of 0. Similar lessons may be skipped.
- Lesson cleanup runs during training, not during feedback.

Lessons and their scores are stored under `storage/memory_db/`.

## API

The API is intended for local development and is available at
`http://127.0.0.1:8000`.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/train?rounds=5` | Run training and save the results. |
| `GET` | `/results` | Read saved training results. Returns 404 before the first training run. |
| `GET` | `/results?ensure_before=true` | Also calculate and save the baseline test score if it has not been calculated yet. |
| `POST` | `/classify` | Classify a message. `use_memory` and `use_best_prompt` default to `true`. |
| `POST` | `/feedback` | Reinforce or correct lessons based on a user's feedback. |
| `GET` | `/lessons` | List lessons and their scores. |

Example classification request:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Match payment 9912 to invoice INV-88",
    "use_memory": true,
    "use_best_prompt": true
  }'
```

Example feedback request:

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Who matched payment 9912 to invoice INV-88?",
    "predicted": "match_invoice",
    "correct": "audit_query"
  }'
```

## Data and generated files

Each dataset is a JSON array of objects with `text` and `intent` fields.

| File | Purpose |
|---|---|
| `data/train.json` | Examples used for training and prompt optimization. |
| `data/test.json` | Held-out evaluation after training and baseline evaluation. |
| `data/demo.json` | Examples displayed in the UI's Demo set. |

Training results and persistent memory are generated under `storage/`:

- `storage/results.json` contains the best prompt and evaluation results.
- `storage/memory_db/` contains the ChromaDB lesson collection.

These are local runtime data, not source files.

## Inspect or reset memory

Print a read-only report about the local ChromaDB data:

```bash
uv run python scripts/inspect_chroma.py
```

To reset memory, stop the API, UI, and any training process, then remove the
database directory:

```bash
rm -rf storage/memory_db
```

The database is recreated when the application next uses it. To also remove
the saved prompt and training results:

```bash
rm -f storage/results.json
```

## Development

Install development dependencies and run the checks:

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
```

## Project layout

```text
src/intent_agent/
  agent/       Classification and intent labels
  api/         FastAPI endpoints
  evaluation/  Dataset evaluation and lesson score updates
  memory/      ChromaDB lesson storage and retrieval
  optimizer/   Prompt rewrites and lesson generation
  training/    Batch training loop
  ui/          Streamlit application
  config.py    Environment settings and project paths
  llm.py       OpenRouter client
data/          Training, test, and demo datasets
scripts/       ChromaDB inspection utility
storage/       Generated training results and local memory
tests/         Import and debug-output tests
```
