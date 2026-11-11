# Contributing & Git

## Before you push

From the project root:

```bash
uv sync --group dev
uv run ruff check .
```

Optional (auto-format style only; run when you want a cleanup PR):

```bash
uv run ruff format .
uv run ruff format --check .
```

Optional: run the app smoke test (API must be configured via `.env`):

```bash
uv run python -c "import api, agent, llm"
```

## What to commit

| Include | Examples |
|--------|-----------|
| Source | `*.py`, `data/*.json`, `README.md`, `CONTRIBUTING.md` |
| Project config | `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore` |
| CI | `.github/workflows/*.yml` |

## Never commit

| Exclude | Why |
|---------|-----|
| `.env` | API keys |
| `memory_db/` | Local Chroma data |
| `results.json` | Generated after training |
| `.venv/` | Virtualenv |

If something was committed by mistake:

```bash
git rm -r --cached memory_db .env results.json 2>/dev/null
git commit -m "Stop tracking local runtime files"
```

## Git workflow (simple)

1. Create a branch: `git checkout -b your-feature`
2. Change code; run **ruff** (above)
3. Stage only intended files: `git status` then `git add <files>`
4. Commit: `git commit -m "Short why-focused message"`
5. Push: `git push -u origin your-feature`
6. Open a pull request on GitHub

## Commit messages

- One line summary (imperative): `Add before/after score to training tab`
- Optional body: what and why, not a file list

## First-time GitHub upload

```bash
git init
git add .
git status   # confirm no .env, memory_db/, results.json
git commit -m "Initial intent-agent sample"
git remote add origin https://github.com/<org>/<repo>.git
git branch -M main
git push -u origin main
```

Use `git status` and `git check-ignore -v <path>` if unsure whether a file is ignored.
