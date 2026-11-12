#!/usr/bin/env python3
"""Read-only Chroma inspector for this project. Does not modify memory_db."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb

from intent_agent.config import MEMORY_DB_DIR

PATH = Path(MEMORY_DB_DIR)
COLLECTION = "lessons"


def main() -> None:
  if not PATH.is_dir():
    print(f"No folder at {PATH.resolve()} — run training or chat first.")
    sys.exit(1)

  print("=== On disk ===")
  for p in sorted(PATH.rglob("*")):
    rel = p.relative_to(PATH)
    if p.is_dir():
      print(f"  [dir]  {rel}/")
    else:
      print(f"  [file] {rel}  ({p.stat().st_size} bytes)")

  print("\n=== Chroma client ===")
  db = chromadb.PersistentClient(path=str(PATH))
  collections = db.list_collections()
  print(f"  path: {PATH.resolve()}")
  print(f"  collections ({len(collections)}): {[c.name for c in collections]}")

  names = {c.name for c in collections}
  if COLLECTION not in names:
    print(f"  collection '{COLLECTION}' not found (empty DB).")
    sys.exit(0)

  col = db.get_collection(COLLECTION)
  meta = col.metadata
  count = col.count()
  print(f"\n=== Collection '{COLLECTION}' ===")
  print(f"  count: {count}")
  print(f"  metadata: {meta}")

  if count == 0:
    print("  (no documents)")
    sys.exit(0)

  data = col.get(include=["documents", "metadatas", "embeddings"])
  ids = data["ids"]
  docs = data["documents"]
  metas = data["metadatas"]
  embs = data.get("embeddings")

  print("\n=== Documents ===")
  for i, doc_id in enumerate(ids):
    score = metas[i].get("score") if metas and metas[i] else None
    doc = docs[i] if docs else ""
    emb_dim = "n/a"
    if embs is not None and i < len(embs) and embs[i] is not None:
      emb_dim = len(embs[i])
    print(f"--- id: {doc_id}")
    print(f"    score: {score}")
    print(f"    embedding_dim: {emb_dim}")
    text = doc if len(doc) <= 200 else doc[:200] + "..."
    print(f"    document: {text!r}")

  print("\n=== Example query (top 3 by text) ===")
  sample = docs[0] if docs else "test"
  r = col.query(query_texts=[sample], n_results=min(3, count))
  print(json.dumps(
    {
      "query": sample,
      "ids": r["ids"],
      "distances": r["distances"],
      "documents": r["documents"],
    },
    indent=2,
  ))


if __name__ == "__main__":
  main()
