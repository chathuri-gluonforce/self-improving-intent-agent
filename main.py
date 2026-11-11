import json
import memory
from evaluate import evaluate
from optimizer import new_prompt, make_lesson

def run_training(rounds=5):
    train = json.load(open("data/train.json"))
    test = json.load(open("data/test.json"))
    prompt = "Classify the customer message intent."
    history, log = [], []

    for r in range(1, rounds + 1):
        score, mistakes = evaluate(train, prompt, learn=True)
        history.append((prompt, score))
        for m in mistakes[:3]:
            memory.add(make_lesson(m))
        memory.cleanup()
        log.append({"round": r, "score": score, "prompt": prompt,
                    "lessons": len(memory.all_lessons())})
        print(f"Round {r}: score {score:.2f}")
        prompt = new_prompt(history, mistakes)

    best = max(history, key=lambda x: x[1])[0]
    test_score, _ = evaluate(test, best)
    result = {"log": log, "best_prompt": best, "test_score": test_score}
    json.dump(result, open("results.json", "w"), indent=2)
    return result

if __name__ == "__main__":
    run_training()