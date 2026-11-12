#gives the score, which is the learning signal
from agent import classify
import memory

def evaluate(data, prompt, learn=False, use_memory=True):
    correct, mistakes = 0, []
    for ex in data:
        pred, used = classify(ex["text"], prompt, use_memory=use_memory)
        ok = pred == ex["intent"]
        correct += ok
        if learn:
            for mid in used:
                memory.update(mid, 1 if ok else -1)
        if not ok:
            mistakes.append({**ex, "pred": pred})
    return correct / len(data), mistakes