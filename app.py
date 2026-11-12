import json

import pandas as pd
import requests
import streamlit as st

from agent import DEFAULT_PROMPT, LABELS

API = "http://127.0.0.1:8000"

PRETTY = {
    "match_invoice": "Match invoice",
    "unmatched_items": "Unmatched items",
    "upload_document": "Upload document",
    "amount_mismatch": "Amount mismatch",
    "recon_status": "Recon status",
    "user_access": "User access",
    "audit_query": "Audit log",
    "login_help": "Login help",
    "other": "Other",
}


def pretty(label: str) -> str:
    return PRETTY.get(label, label)


def load_results(ensure_before: bool = False):
    try:
        timeout = 900 if ensure_before else 15
        r = requests.get(
            f"{API}/results",
            params={"ensure_before": ensure_before},
            timeout=timeout,
        )
    except requests.RequestException:
        return None, "unreachable"
    if r.status_code == 404:
        return None, "missing"
    r.raise_for_status()
    return r.json(), None


def classify(text: str, use_memory: bool, use_best_prompt: bool):
    r = requests.post(
        f"{API}/classify",
        json={
            "text": text,
            "use_memory": use_memory,
            "use_best_prompt": use_best_prompt,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def send_feedback(text: str, predicted: str, correct: str):
    r = requests.post(
        f"{API}/feedback",
        json={"text": text, "predicted": predicted, "correct": correct},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def intent_badge(intent: str, correct: str | None):
    name = pretty(intent)
    if correct:
        ok = intent == correct
        bg = "#166534" if ok else "#991b1b"
        name = f"{'Right' if ok else 'Wrong'} · {name}"
    else:
        bg = "#1e3a5f"
    st.markdown(
        (
            "<div style='display:inline-block;padding:0.7rem 1.1rem;"
            "border-radius:999px;color:white;font-size:1.35rem;font-weight:600;"
            f"background:{bg}'>{name}</div>"
        ),
        unsafe_allow_html=True,
    )


def mark(pred: str, correct: str) -> str:
    return f"{'✅' if pred == correct else '❌'} {pretty(pred)}"


def bubble(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content)


def fetch_lessons():
    try:
        r = requests.get(f"{API}/lessons", timeout=10)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return r.json()


def render_sidebar(trained_ready: bool) -> str:
    st.sidebar.header("Agent")
    mode = st.sidebar.radio(
        "Which agent replies in chat",
        ["Normal", "Self-improving"],
    )
    if mode == "Normal":
        st.sidebar.caption("Original prompt. No lessons.")
    else:
        st.sidebar.caption("Trained prompt. Similar lessons are retrieved.")
        if not trained_ready:
            st.sidebar.warning("Run training first.")
    return mode


def paint_lessons(slot):
    """Read lessons after any score change so the sidebar never keeps the old number."""
    items = fetch_lessons()
    with slot.container():
        st.markdown("**Lessons**")
        st.caption("Rules stored in memory")
        if items is None:
            st.warning("Memory is not available. Is the API running?")
        elif not items:
            st.info("No lessons yet.")
        else:
            for item in sorted(items, key=lambda row: row["score"], reverse=True):
                st.markdown(item["lesson"])
                st.caption(f"score: {item['score']}")


st.set_page_config(page_title="Self-Improving Intent Agent", page_icon="💬", layout="wide")
st.title("Self-Improving Intent Agent")
st.caption("Same frozen model. Prompt and memory improve.")

if "chat_normal" not in st.session_state:
    st.session_state.chat_normal = []
if "chat_trained" not in st.session_state:
    st.session_state.chat_trained = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "busy" not in st.session_state:
    st.session_state.busy = None

_sidebar_results, _sidebar_status = load_results()
chat_mode = render_sidebar(_sidebar_status != "missing" and _sidebar_results is not None)
lesson_slot = st.sidebar.empty()
paint_lessons(lesson_slot)

tab_chat, tab_compare, tab_demo, tab_train = st.tabs(
    ["Chat", "Before vs After", "Demo set", "Training results"]
)

with tab_chat:
    trained = chat_mode == "Self-improving"
    history_key = "chat_trained" if trained else "chat_normal"
    if trained:
        st.caption("Self-improving agent. Trained prompt and memory. Yes or No updates lessons.")
    else:
        st.caption("Normal agent. Original prompt and no memory. Feedback is off.")
    if trained and _sidebar_status == "missing":
        st.info("Run training first. This chat will use the original prompt until then.")

    use_trained = trained and _sidebar_results is not None
    busy = st.session_state.busy
    with st.container(height=480, border=True):
        if not st.session_state[history_key] and not busy:
            bubble("assistant", "Hello. Send a customer message and I will name the intent.")
        for message in st.session_state[history_key]:
            bubble(message["role"], message["content"])
        if busy:
            with st.chat_message("assistant"):
                with st.spinner("Typing..."):
                    kind = busy["kind"]
                    if kind == "reply":
                        try:
                            out = classify(
                                busy["text"],
                                use_memory=busy["use_trained"],
                                use_best_prompt=busy["use_trained"],
                            )
                            reply = f"**{pretty(out['intent'])}**"
                            used = out.get("lessons") or []
                            if busy["use_trained"] and used:
                                lines = "\n".join(
                                    f"- {item['lesson']} (score {item['score']})" for item in used
                                )
                                reply += "\n\nLessons used:\n" + lines
                            elif busy["use_trained"]:
                                reply += "\n\nNo similar lessons retrieved."
                            if busy["use_trained"]:
                                st.session_state.pending = {
                                    "text": busy["text"],
                                    "pred": out["intent"],
                                    "awaiting_label": False,
                                }
                        except requests.RequestException:
                            reply = "I could not reach the API. Is uvicorn running?"
                    elif kind == "yes":
                        try:
                            send_feedback(busy["text"], busy["pred"], busy["pred"])
                            reply = "Thanks. That guess was right, so no new lesson was added."
                        except requests.RequestException:
                            reply = "I could not save that feedback. Is the API running?"
                        st.session_state.pending = None
                    else:
                        try:
                            out = send_feedback(busy["text"], busy["pred"], busy["label"])
                            reply = f"Got it. You meant {pretty(busy['label'])}."
                            if out.get("lesson"):
                                reply += f" I learned this: {out['lesson']}"
                        except requests.RequestException:
                            reply = "I could not save that feedback. Is the API running?"
                        st.session_state.pending = None
                st.markdown(reply)
            st.session_state[busy["history_key"]].append({"role": "assistant", "content": reply})
            st.session_state.busy = None
            paint_lessons(lesson_slot)
            st.rerun()

        pending = st.session_state.pending if use_trained and not st.session_state.busy else None
        if pending and not pending.get("awaiting_label"):
            st.write("Is that right?")
            yes_col, no_col, _pad = st.columns([1, 1, 2])
            if yes_col.button("Yes", use_container_width=True):
                st.session_state.busy = {
                    "kind": "yes",
                    "history_key": "chat_trained",
                    "text": pending["text"],
                    "pred": pending["pred"],
                }
                st.rerun()
            if no_col.button("No", use_container_width=True):
                st.session_state.chat_trained.append({
                    "role": "assistant",
                    "content": "What did you mean?",
                })
                st.session_state.pending["awaiting_label"] = True
                st.rerun()

        if pending and pending.get("awaiting_label"):
            st.caption("Pick the correct intent.")
            choice_cols = st.columns([1, 1, 1, 1])
            for i, label in enumerate(LABELS):
                if choice_cols[i % 3].button(
                    pretty(label), key=f"fix_{label}", use_container_width=True
                ):
                    st.session_state.busy = {
                        "kind": "fix",
                        "history_key": "chat_trained",
                        "text": pending["text"],
                        "pred": pending["pred"],
                        "label": label,
                    }
                    st.rerun()

    busy_now = st.session_state.busy is not None
    with st.form("chat_form", clear_on_submit=True, border=False):
        in_col, send_col = st.columns([6, 1], vertical_alignment="bottom")
        typed = in_col.text_input(
            "Message",
            placeholder="Type a customer message and press Send",
            label_visibility="collapsed",
            disabled=busy_now,
        )
        sent = send_col.form_submit_button("Send", type="primary", disabled=busy_now, use_container_width=True)
    user_text = typed.strip() if sent else ""
    if user_text and st.session_state.busy is None:
        st.session_state[history_key].append({"role": "user", "content": user_text})
        st.session_state.pending = None
        st.session_state.busy = {
            "kind": "reply",
            "history_key": history_key,
            "text": user_text,
            "use_trained": use_trained,
        }
        st.rerun()

with tab_compare:
    st.write(
        "Before uses the original prompt and no lessons. "
        "After uses the trained prompt and similar lessons from memory."
    )
    results, status = load_results()
    if status == "unreachable":
        st.error("Could not reach the API. Start it with uvicorn.")
    elif status == "missing":
        st.info("Run training first.")

    text = st.text_input("Customer message", placeholder="Type a hard message")
    correct = st.selectbox(
        "Correct intent (optional)",
        options=[""] + LABELS,
        format_func=lambda x: "Not set" if x == "" else pretty(x),
    )
    if st.button("Classify", type="primary", disabled=not text.strip()):
        try:
            before = classify(text.strip(), use_memory=False, use_best_prompt=False)
            after = None
            if results:
                after = classify(text.strip(), use_memory=True, use_best_prompt=True)
            st.session_state.compare = {"text": text.strip(), "before": before, "after": after}
        except requests.RequestException as exc:
            st.error(f"Classify failed: {exc}")

    shown = st.session_state.get("compare")
    if shown and shown["text"] == text.strip() and text.strip():
        truth = correct or None
        left, right = st.columns(2)
        with left:
            st.subheader("Before")
            st.caption("Default prompt. No memory.")
            intent_badge(shown["before"]["intent"], truth)
            st.text_area(
                "Prompt",
                shown["before"].get("prompt", DEFAULT_PROMPT),
                height=140,
                disabled=True,
                key="before_prompt",
            )
        with right:
            st.subheader("After")
            if shown["after"] is None:
                st.info("Run training first.")
            else:
                st.caption("Trained prompt. Lessons retrieved.")
                intent_badge(shown["after"]["intent"], truth)
                st.text_area(
                    "Prompt",
                    shown["after"].get("prompt", ""),
                    height=140,
                    disabled=True,
                    key="after_prompt",
                )
                st.markdown("**Lessons used**")
                lessons = shown["after"].get("lessons") or []
                if not lessons:
                    st.caption("No similar lessons retrieved.")
                for item in lessons:
                    st.write(item["lesson"])
                    st.caption(f"score: {item['score']}")

with tab_demo:
    st.write("Same hard messages, scored twice: before training and after training.")
    try:
        demo = json.load(open("data/demo.json"))
    except OSError:
        demo = []
        st.error("data/demo.json is missing.")

    results, status = load_results()
    if status == "unreachable":
        st.error("Could not reach the API. Start it with uvicorn.")
    elif status == "missing":
        st.info("Run training first.")

    if st.button("Run demo", type="primary", disabled=not demo or not results):
        rows = []
        progress = st.progress(0.0, text="Running demo")
        try:
            for i, ex in enumerate(demo, start=1):
                before = classify(ex["text"], use_memory=False, use_best_prompt=False)
                after = classify(ex["text"], use_memory=True, use_best_prompt=True)
                rows.append(
                    {
                        "message": ex["text"],
                        "correct": pretty(ex["intent"]),
                        "before guess": mark(before["intent"], ex["intent"]),
                        "after guess": mark(after["intent"], ex["intent"]),
                        "_before_ok": before["intent"] == ex["intent"],
                        "_after_ok": after["intent"] == ex["intent"],
                    }
                )
                progress.progress(i / len(demo), text=f"{i} / {len(demo)}")
            st.session_state.demo_rows = rows
        except requests.RequestException as exc:
            st.error(f"Demo failed: {exc}")
        finally:
            progress.empty()

    rows = st.session_state.get("demo_rows") or []
    if rows:
        n = len(rows)
        before_n = sum(row["_before_ok"] for row in rows)
        after_n = sum(row["_after_ok"] for row in rows)
        m1, m2 = st.columns(2)
        m1.metric("Before", f"{before_n}/{n} correct")
        m2.metric("After", f"{after_n}/{n} correct")
        table = pd.DataFrame(rows).drop(columns=["_before_ok", "_after_ok"])
        st.dataframe(table, use_container_width=True, hide_index=True)

with tab_train:
    st.write("Saved training run. Nothing here calls the optimizer unless you press the button at the bottom.")
    results, status = load_results()
    if status == "unreachable":
        st.error("Could not reach the API. Start it with uvicorn.")
    elif status == "missing":
        st.info("Run training first.")
    else:
        log = pd.DataFrame(results.get("log") or [])
        if not log.empty:
            st.subheader("Train score each round")
            st.line_chart(log.set_index("round")["score"])

        if "test_score_before" not in results and st.button("Compute before score"):
            with st.spinner("Scoring the original prompt on the test set. This runs once."):
                results, status = load_results(ensure_before=True)
            if results is None:
                st.error("Could not compute the before score.")
                results = {}

        before_score = results.get("test_score_before")
        after_score = results.get("test_score")
        c1, c2 = st.columns(2)
        c1.metric(
            "Test score before",
            "—" if before_score is None else f"{before_score * 100:.0f}%",
        )
        c2.metric(
            "Test score after",
            "—" if after_score is None else f"{after_score * 100:.0f}%",
        )
        st.caption(
            "Test set. Before is the original prompt with no memory. "
            "After is the saved best prompt. Compute before once; it is saved in results.json."
        )

        st.subheader("Prompt adjustments during training")
        st.caption("Each round is the prompt used, then its train score. The next round is the rewritten prompt.")
        if log.empty:
            st.caption("No rounds saved.")
        else:
            for _, row in log.iterrows():
                title = (
                    f"Round {int(row['round'])} · score {row['score'] * 100:.0f}%"
                    f" · {int(row['lessons'])} lessons"
                )
                with st.expander(title, expanded=int(row["round"]) in (1, len(log))):
                    st.write(row["prompt"])

        st.subheader("First prompt and best prompt")
        first = log.iloc[0]["prompt"] if not log.empty else DEFAULT_PROMPT
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**First prompt**")
            st.text_area("first", first, height=160, disabled=True, label_visibility="collapsed")
        with p2:
            st.markdown("**Best prompt**")
            st.text_area(
                "best",
                results.get("best_prompt", ""),
                height=160,
                disabled=True,
                label_visibility="collapsed",
            )

        st.subheader("Lessons in memory")
        try:
            lesson_resp = requests.get(f"{API}/lessons", timeout=15)
            lesson_resp.raise_for_status()
            lessons = lesson_resp.json()
        except requests.RequestException:
            lessons = None
            st.warning("Could not load lessons.")
        if lessons is not None:
            if not lessons:
                st.caption("No lessons yet.")
            else:
                frame = pd.DataFrame(lessons).sort_values("score", ascending=False)
                st.dataframe(frame, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("Run training"):
        with st.spinner("Training. This can take several minutes."):
            try:
                requests.post(f"{API}/train", timeout=900)
            except requests.RequestException:
                st.error("Could not start training. Is the API running?")
            else:
                st.session_state.pop("compare", None)
                st.session_state.pop("demo_rows", None)
                st.session_state.chat_normal = []
                st.session_state.chat_trained = []
                st.session_state.pending = None
                st.session_state.busy = None
                paint_lessons(lesson_slot)
                st.rerun()
