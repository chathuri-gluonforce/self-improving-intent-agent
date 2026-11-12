import uuid
import chromadb

db = chromadb.PersistentClient(path="memory_db")
col = db.get_or_create_collection("lessons")

def add(text):                      # Create
    if col.count() > 0:
        r = col.query(query_texts=[text], n_results=1)
        #closest, similar meaning lessons ignore in here. not save again. 
        if r["distances"][0][0] < 0.2:   # very similar lesson already saved
            return
    #Chroma runs its default embedder (MiniLM, 384-d) on that text → vector.
    col.add(ids=[str(uuid.uuid4())], documents=[text], metadatas=[{"score": 0}])
    print(f"saved: score=0  {text}")

def search(query, k=3):             # Read
    if col.count() == 0:
        return []
    r = col.query(query_texts=[query], n_results=min(k, col.count()))
    return list(zip(r["ids"][0], r["documents"][0]))

def update(mem_id, reward):         # Update
    m = col.get(ids=[mem_id])["metadatas"][0]
    col.update(ids=[mem_id], metadatas=[{"score": m["score"] + reward}])

def cleanup(min_score=-2):          # Delete
    a = col.get()
    bad = [i for i, m in zip(a["ids"], a["metadatas"]) if m["score"] <= min_score]
    if bad:
        col.delete(ids=bad)

def all_lessons():
    a = col.get()
    for d, m in zip(a["documents"], a["metadatas"]):
        print(f"score={m['score']}  {d}")
    return [(d, m["score"]) for d, m in zip(a["documents"], a["metadatas"])]


def lessons_by_ids(ids):
    """Read lesson text and score for ids already returned by search. Does not write."""
    if not ids:
        return []
    found = col.get(ids=list(ids))
    by_id = {
        item_id: {"lesson": doc, "score": meta["score"]}
        for item_id, doc, meta in zip(found["ids"], found["documents"], found["metadatas"])
    }
    return [{"lesson": by_id[item_id]["lesson"], "score": by_id[item_id]["score"]}
            for item_id in ids if item_id in by_id]