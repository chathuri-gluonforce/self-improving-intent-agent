from llm import ask
import memory

LABELS = ["match_invoice", "unmatched_items", "upload_document", "amount_mismatch", "recon_status", "user_access", "audit_query", "login_help", "other"]

def classify(text, prompt):
    lessons = memory.search(text)
    notes = "\n".join(f"- {l}" for _, l in lessons) or "- none"
    full = (f"{prompt}\nLabels: {', '.join(LABELS)}\n"
            f"Lessons from past mistakes:\n{notes}\n"
            f"Message: {text}\nAnswer with one label only.")
    return ask(full).lower().strip(), [i for i, _ in lessons]