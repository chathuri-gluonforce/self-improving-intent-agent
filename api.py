import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import classify
from main import run_training
from optimizer import make_lesson
import memory

app = FastAPI(title="Self-Improving Intent Agent")

class Message(BaseModel):
    text: str

class Feedback(BaseModel):
    text: str
    predicted: str
    correct: str

def load_results():
    if not os.path.exists("results.json"):
        raise HTTPException(404, "No results yet. Call POST /train first.")
    return json.load(open("results.json"))

def get_prompt():
    if os.path.exists("results.json"):
        return json.load(open("results.json"))["best_prompt"]
    return "Classify the customer message intent."

@app.post("/train")
def train(rounds: int = 5):
    return run_training(rounds)

@app.get("/results")
def results():
    return load_results()

@app.post("/classify")
def classify_msg(msg: Message):
    intent, _ = classify(msg.text, get_prompt())
    return {"text": msg.text, "intent": intent}

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
    return [{"lesson": l, "score": s} for l, s in memory.all_lessons()]