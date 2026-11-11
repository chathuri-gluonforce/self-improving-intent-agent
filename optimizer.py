import re
from llm import ask
from agent import LABELS

def new_prompt(history, mistakes):
    def parts(item):
        if isinstance(item, dict):
            return item["prompt"], item["score"]
        return item[0], item[1]

    hist = "\n".join(
        f"score={s:.3f}\nprompt: {p}"
        for p, s in sorted((parts(h) for h in history), key=lambda x: x[1])
    )
    label_list = ", ".join(LABELS)
    errs = "\n".join(
        f"'{m['text']}' -> said {m['pred']}, correct {m['intent']}"
        for m in mistakes[:5]
    ) or "None"
    meta = (
        f"Write a better classification prompt.\n\n"
        f"Rules:\n"
        f"- Use ONLY these intents: {label_list}\n"
        f"- Do not invent new intents.\n"
        f"- Tell the model to answer with one label only, no explanation.\n"
        f"- Keep the prompt short and clear.\n"
        f"- Do not include scores, tags, or comments in the prompt.\n"
        f"- Put only the prompt inside <PROMPT> and </PROMPT>.\n\n"
        f"Previous prompts and their scores (low to high):\n{hist}\n\n"
        f"Some mistakes:\n{errs}\n"
    )
    out = ask(meta, temperature=0.5)
    found = re.search(r"<PROMPT>(.*?)</PROMPT>", out, re.S | re.I)
    if not found:
        return parts(history[-1])[0] if history else "Classify the customer message intent."
    p = found.group(1).strip()
    if "score:" in p.lower() or "rewritten prompt" in p.lower():
        return parts(history[-1])[0]
    return p

def make_lesson(m):
    return ask(
        f"Message: '{m['text']}'. Model said '{m['pred']}', "
        f"correct is '{m['intent']}'. Write one short general rule "
        "to avoid this mistake."
    )