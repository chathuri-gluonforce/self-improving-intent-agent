import requests
import pandas as pd
import streamlit as st

API = "http://127.0.0.1:8000"

QUESTIONS = {
    "match_invoice": "match a bank payment to an invoice",
    "unmatched_items": "see unmatched items",
    "upload_document": "upload a document",
    "amount_mismatch": "flag an amount mismatch",
    "recon_status": "check reconciliation status",
    "user_access": "manage user access",
    "audit_query": "check the audit log",
    "login_help": "get help logging in",
    "other": "something else",
}
CHOICES = [
    ("match_invoice", "Match invoice"),
    ("unmatched_items", "Unmatched items"),
    ("upload_document", "Upload document"),
    ("amount_mismatch", "Amount mismatch"),
    ("recon_status", "Recon status"),
    ("user_access", "User access"),
    ("audit_query", "Audit log"),
    ("login_help", "Login help"),
    ("other", "Other"),
]

st.set_page_config(page_title="Self-Improving Intent Agent", page_icon="💬")
st.title("Self-Improving Intent Agent")
st.write(
    "This app classifies customer messages. The model stays the same. "
    "It improves from training data, and from your yes or no feedback."
)
st.write("**Intents we have:** Match invoice · Unmatched items · Upload document · Amount mismatch · Recon status · User access · Audit log · Login help · Other")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "pending" not in st.session_state:
    st.session_state.pending = None

def ask_agent(text):
    r = requests.post(f"{API}/classify", json={"text": text}, timeout=60)
    r.raise_for_status()
    return r.json()["intent"]

def send_feedback(text, predicted, correct):
    r = requests.post(
        f"{API}/feedback",
        json={"text": text, "predicted": predicted, "correct": correct},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()

def fetch_lessons():
    try:
        r = requests.get(f"{API}/lessons", timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return None

def render_sidebar_lessons(items):
    st.sidebar.header("Memory")
    st.sidebar.caption("Lessons the agent has learned")
    if items is None:
        st.sidebar.warning("Memory is not available. Is the API running?")
    elif items:
        for item in items:
            st.sidebar.markdown(item["lesson"])
            st.sidebar.caption(f"score: {item['score']}")
    else:
        st.sidebar.info("No lessons yet.")

tab_talk, tab_train = st.tabs(["Talk to the agent", "Training and memory"])

with tab_talk:
    st.caption(
        "Type a customer message. The agent guesses, then asks you to confirm. "
        "Only **No** (wrong guess) adds a new lesson; **Yes** just reinforces lessons already used."
    )

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    pending = st.session_state.pending
    if pending and not pending.get("awaiting_label"):
        c1, c2 = st.columns(2)
        if c1.button("Yes", use_container_width=True):
            send_feedback(pending["text"], pending["pred"], pending["pred"])
            st.session_state.chat.append({
                "role": "assistant",
                "content": "Thanks — got it.",
            })
            st.session_state.pending = None
            st.rerun()
        if c2.button("No", use_container_width=True):
            st.session_state.chat.append({
                "role": "assistant",
                "content": "Sorry about that. What did you mean?",
            })
            st.session_state.pending["awaiting_label"] = True
            st.rerun()

    if pending and pending.get("awaiting_label"):
        cols = st.columns(len(CHOICES))
        for i, (key, label) in enumerate(CHOICES):
            if cols[i].button(label, use_container_width=True):
                out = send_feedback(pending["text"], pending["pred"], key)
                lesson = out.get("lesson")
                reply = f"Got it. You meant {label}."
                if lesson:
                    reply += f" I learned this: {lesson}"
                st.session_state.chat.append({"role": "assistant", "content": reply})
                st.session_state.pending = None
                st.rerun()

    user_text = st.chat_input("Type a customer message")
    if user_text:
        st.session_state.chat.append({"role": "user", "content": user_text})
        try:
            pred = ask_agent(user_text)
            question = QUESTIONS.get(pred, f"I think this is {pred}. Is that right?")
            st.session_state.chat.append({
                "role": "assistant",
                "content": f"Intent: **{pred}**\n\n{question}",
            })
            st.session_state.pending = {
                "text": user_text,
                "pred": pred,
                "awaiting_label": False,
            }
        except Exception:
            st.session_state.chat.append({
                "role": "assistant",
                "content": "I could not reach the backend. Is uvicorn running?",
            })
            st.session_state.pending = None
        st.rerun()

with tab_train:
    if st.button("Run training"):
        with st.spinner("Training..."):
            try:
                requests.post(f"{API}/train", timeout=600)
            except Exception:
                st.error("Could not start training. Is the API running?")
            else:
                st.rerun()

    try:
        r = requests.get(f"{API}/results", timeout=10)
    except Exception:
        st.warning("I could not reach the backend. Is uvicorn running?")
        r = None

    if r is not None and r.status_code == 200:
        res = r.json()
        df = pd.DataFrame(res["log"])
        st.subheader("How the score changed")
        st.line_chart(df.set_index("round")["score"])
        st.metric("Final test score", f"{res['test_score']:.2f}")
        st.subheader("How the prompt changed")
        st.dataframe(df[["round", "score", "prompt"]])
        st.subheader("Lessons in memory")
        lesson_rows = fetch_lessons()
        if lesson_rows is not None:
            st.table(pd.DataFrame(lesson_rows))
        else:
            st.warning("Could not load lessons from the API.")
    else:
        st.info("No training results yet. Click Run training, or teach the agent in the chat tab.")

# After tabs (and after training may have run): sidebar sees the same memory as the train tab.
render_sidebar_lessons(fetch_lessons())