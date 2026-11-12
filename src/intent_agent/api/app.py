import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from intent_agent.agent.classify import DEFAULT_PROMPT, classify
from intent_agent.config import RESULTS_FILE, TEST_FILE
from intent_agent.evaluation.evaluate import evaluate
from intent_agent.memory import store as memory
from intent_agent.optimizer.opro import make_lesson
from intent_agent.training.train import run_training

app = FastAPI(title="Self-Improving Intent Agent")


class Message(BaseModel):
    text: str
    use_memory: bool = True
    use_best_prompt: bool = True


class Feedback(BaseModel):
    text: str
    predicted: str
    correct: str


def load_results():
    if not os.path.exists(RESULTS_FILE):
        raise HTTPException(404, "No results yet. Call POST /train first.")
    with open(RESULTS_FILE) as f:
        return json.load(f)


def save_results(data):
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_prompt():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)["best_prompt"]
    return DEFAULT_PROMPT


def resolve_prompt(use_best_prompt: bool) -> str:
    if use_best_prompt and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)["best_prompt"]
    return DEFAULT_PROMPT


def ensure_before_score(data):
    """Score the default prompt with no memory once, then cache it."""
    if "test_score_before" in data:
        return data
    with open(TEST_FILE) as f:
        test = json.load(f)
    score, _ = evaluate(test, DEFAULT_PROMPT, learn=False, use_memory=False)
    data["test_score_before"] = score
    save_results(data)
    return data


@app.post("/train")
def train(rounds: int = 5):
    return run_training(rounds)


@app.get("/results")
def results(ensure_before: bool = False):
    data = load_results()
    if ensure_before:
        data = ensure_before_score(data)
    return data


@app.post("/classify")
def classify_msg(msg: Message):
    prompt = resolve_prompt(msg.use_best_prompt)
    intent, used_ids = classify(msg.text, prompt, use_memory=msg.use_memory)
    lessons = memory.lessons_by_ids(used_ids) if msg.use_memory else []
    return {
        "text": msg.text,
        "intent": intent,
        "prompt": prompt,
        "lessons": lessons,
    }


@app.post("/feedback")
def feedback(fb: Feedback):
    _, used = classify(fb.text, get_prompt())
    if fb.correct == fb.predicted:
        for mid in used:
            memory.update(mid, 1)
        return {"learned": "confirmed"}
    for mid in used:
        memory.update(mid, -1)
    lesson = make_lesson({"text": fb.text, "pred": fb.predicted, "intent": fb.correct})
    memory.add(lesson)
    return {"learned": "corrected", "lesson": lesson}


@app.get("/lessons")
def lessons():
    return [{"lesson": lesson, "score": score} for lesson, score in memory.all_lessons()]
