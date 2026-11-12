import json
from intent_agent.config import RESULTS_FILE, TEST_FILE, TRAIN_FILE
from intent_agent.memory import store as memory
from intent_agent.evaluation.evaluate import evaluate
from intent_agent.optimizer.opro import new_prompt, make_lesson

def run_training(rounds=5):
    train = json.load(open(TRAIN_FILE))
    test = json.load(open(TEST_FILE))
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
    json.dump(result, open(RESULTS_FILE, "w"), indent=2)
    return result

if __name__ == "__main__":
    run_training()