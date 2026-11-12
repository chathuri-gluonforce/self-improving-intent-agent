from llm import ask
import memory

LABELS = [
    "match_invoice",
    "unmatched_items",
    "upload_document",
    "amount_mismatch",
    "recon_status",
    "user_access",
    "audit_query",
    "login_help",
    "other",
]

DEFAULT_PROMPT = "Classify the customer message intent."


def classify(text, prompt, use_memory=True):
    lessons = memory.search(text) if use_memory else []
    notes = "\n".join(f"- {lesson}" for _, lesson in lessons) or "- none"
    full = (
        f"{prompt}\nLabels: {', '.join(LABELS)}\n"
        f"Lessons from past mistakes:\n{notes}\n"
        f"Message: {text}\nAnswer with one label only."
    )
    pred = ask(full).lower().strip()
    return pred, [item_id for item_id, _ in lessons]
