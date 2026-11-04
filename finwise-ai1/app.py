"""Codex/Ollama assignment: local Gemma chat + document collection."""
from pathlib import Path
import json
import urllib.error
import urllib.request
import streamlit as st

st.set_page_config(page_title="Chat with Gemma | Local AI", page_icon="🌿", layout="centered")
FILES = Path(__file__).resolve().parent / "my_files"
FILES.mkdir(exist_ok=True)
MODEL = "gemma3:1b"
MAX_BYTES = 50 * 1024 * 1024

st.markdown("""
<style>
:root { --green: #5b8b65; --ink: #242c27; }
[data-testid="stAppViewContainer"] { background: #f4f6f3; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
  max-width: 850px; background: white; border: 1px solid #e4eae3;
  border-radius: 19px; padding: 34px 34px 42px; margin-top: 28px;
  box-shadow: 0 2px 12px #132b1908;
}
h1, h2, h3, p, label { color: var(--ink); }
h1 { font-size: 29px !important; font-weight: 760 !important; letter-spacing: -0.8px; margin: 0 !important; }
.kicker { color: #5b8b65; font-weight: 800; letter-spacing: 2px; font-size: 11px; margin: 0 0 4px; }
.intro { color: #7b877d; font-size: 14px; margin: 4px 0 10px; }
[data-baseweb="tab-list"] { gap: 15px; border-bottom: 1px solid #e8ede7; }
button[data-baseweb="tab"] { font-weight: 650; color: #62886b; padding: 12px 0; }
button[data-baseweb="tab"][aria-selected="true"] { color: #42734e; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #5b8b65; }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
 background: #5b8b65 !important; border-color:#5b8b65 !important; color:white !important;
}
.stButton button, [data-testid="stFormSubmitButton"] button { border-radius: 9px; }
[data-testid="stChatMessage"] { background: #f8faf7; border: 1px solid #e6eee5; border-radius: 13px; margin-bottom: 9px; }
[data-testid="stFileUploader"] { border-radius: 12px; }
.smallhint {font-size: 13px; color: #8c968c; }
.chat-empty {height: 210px; display:flex; align-items:center; justify-content:center; color:#8b988e; font-size:14px; border-bottom: 1px solid #e9eee9; margin-bottom: 14px;}
hr { border-color:#e8ede7; }
</style>
""", unsafe_allow_html=True)


def clean_name(filename):
    name = Path(filename).name
    if not name or name in (".", "..") or "\\" in name or "/" in name:
        raise ValueError("Choose a valid file name")
    return name


def ask_ollama(messages):
    payload = json.dumps({"model": MODEL, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.load(response)
    return result["message"]["content"]


if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="kicker">LOCAL AI</div>', unsafe_allow_html=True)
st.title("Chat with Gemma")
st.markdown('<div class="intro">A private conversation with your local Ollama model.</div>', unsafe_allow_html=True)
chat, documents, budget = st.tabs(["Chat", "Document collection", "Monthly budget"])

with chat:
    left, right = st.columns([5, 1.6], vertical_alignment="center")
    with left:
        st.markdown("<p class='smallhint'>Ollama · gemma3:1b · Runs on your Mac</p>", unsafe_allow_html=True)
    with right:
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if not st.session_state.messages:
        st.markdown('<div class="chat-empty">Ask Gemma anything to get started.</div>', unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    with st.form("send_chat", clear_on_submit=True):
        prompt = st.text_input("Your message", placeholder="Ask Gemma anything...")
        sent = st.form_submit_button("Send", type="primary")
    if sent and prompt.strip():
        st.session_state.messages.append({"role":"user", "content": prompt.strip()})
        try:
            with st.spinner("Gemma is thinking..."):
                answer = ask_ollama(st.session_state.messages)
            st.session_state.messages.append({"role":"assistant", "content": answer})
            st.rerun()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                st.error("Gemma model not installed. In Terminal run: ollama pull gemma3:1b")
            else:
                st.error(f"Ollama returned error {exc.code}: {exc.reason}")
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            st.error(f"Cannot connect to Ollama. Open Ollama and run 'ollama pull gemma3:1b' in Terminal. Details: {exc}")
    st.markdown('<div class="smallhint">History stays in this tab while the app is open. New chat clears the conversation.</div>', unsafe_allow_html=True)

with documents:
    st.markdown("### Document collection")
    st.markdown("<p class='intro'>Upload and manage the files you want to keep with your collection.</p>", unsafe_allow_html=True)
    st.markdown("#### Add documents")
    with st.form("upload_documents", clear_on_submit=True):
        uploaded = st.file_uploader("Choose a file", label_visibility="visible")
        save = st.form_submit_button("Upload", type="primary")
    st.caption("Maximum file size: 50 MB. Files stay on your computer and are not sent to Ollama.")
    if save:
        if uploaded is None:
            st.warning("Choose a file first.")
        elif uploaded.size > MAX_BYTES:
            st.error("File exceeds 50 MB.")
        else:
            try:
                name = clean_name(uploaded.name)
                dest = FILES / name
                if dest.exists():
                    st.warning(f"{name} already exists. Rename your file before uploading.")
                else:
                    dest.write_bytes(uploaded.getvalue())
                    st.success(f"Uploaded {name}!")
            except (ValueError, OSError) as exc:
                st.error(str(exc))

    st.markdown("#### Your files")
    files = sorted((p for p in FILES.iterdir() if p.is_file()), key=lambda p: p.name.lower())
    st.caption(f"{len(files)} file(s)")
    if not files:
        st.info("No files yet. Upload a document to start your collection.")
    for idx, path in enumerate(files):
        with st.container(border=True):
            st.markdown(f"**📄 {path.name}**")
            st.caption(f"{path.stat().st_size:,} bytes")
            try:
                st.download_button("Download", data=path.read_bytes(), file_name=path.name, key=f"dl_{idx}")
            except OSError as exc:
                st.error(f"Could not read file: {exc}")
            if st.button("Delete", key=f"delete_{idx}"):
                try:
                    path.unlink()
                    st.rerun()
                except OSError as exc:
                    st.error(f"Could not delete file: {exc}")
    if st.button("Refresh files"):
        st.rerun()

with budget:
    st.markdown("### Your monthly plan")
    st.markdown(
        "<p class='intro'>A simple view of where your baht goes, built for student life.</p>",
        unsafe_allow_html=True,
    )
    income = st.number_input(
        "Monthly income (THB)", min_value=0.0, value=15000.0, step=500.0,
        help="Include allowance, part-time income, or other money you receive this month.",
        key="budget_income",
    )

    st.markdown("#### Monthly expenses")
    expense_columns = st.columns(2)
    expense_fields = [
        ("Food", "expense_food"),
        ("Transportation", "expense_transport"),
        ("Shopping", "expense_shopping"),
        ("Rent", "expense_rent"),
        ("Others", "expense_others"),
    ]
    expenses = {}
    for index, (label, key) in enumerate(expense_fields):
        with expense_columns[index % 2]:
            expenses[label] = st.number_input(
                f"{label} (THB)", min_value=0.0, value=0.0, step=100.0,
                key=key,
            )

    total_expenses = sum(expenses.values())
    remaining = income - total_expenses
    metrics = st.columns(3)
    metrics[0].metric("Monthly income", f"฿{income:,.0f}")
    metrics[1].metric("Total expenses", f"฿{total_expenses:,.0f}")
    metrics[2].metric("Money remaining", f"฿{remaining:,.0f}")
    if total_expenses > income:
        st.warning(f"Expenses are ฿{total_expenses - income:,.0f} over your monthly income.")
    elif income > 0:
        st.success("Your expenses fit within your monthly income.")

    # 50/30/20 guide: essentials are Needs; discretionary purchases are Wants.
    # Remaining money represents Savings for this simple monthly plan.
    needs = expenses["Food"] + expenses["Transportation"] + expenses["Rent"]
    wants = expenses["Shopping"] + expenses["Others"]
    savings = max(remaining, 0)
    actuals = {"Needs": needs, "Wants": wants, "Savings": savings}
    targets = {"Needs": income * 0.50, "Wants": income * 0.30, "Savings": income * 0.20}

    st.markdown("#### 50 / 30 / 20 breakdown")
    st.caption("Guide: 50% for needs, 30% for wants, and 20% for savings. Food, transport, and rent count as needs; shopping and others count as wants.")
    for name, share in (("Needs", 50), ("Wants", 30), ("Savings", 20)):
        target = targets[name]
        actual = actuals[name]
        detail, amount = st.columns([4, 1])
        detail.markdown(f"**{name}** · {share}% target (฿{target:,.0f})")
        amount.markdown(f"**฿{actual:,.0f}**")
        st.progress(min(actual / target, 1.0) if target > 0 else (1.0 if actual > 0 else 0.0))

    st.markdown("#### Spending at a glance")
    chart_data = {"THB": expenses}
    st.bar_chart(chart_data, height=250, color="#5b8b65")
    st.caption("Savings is calculated as income minus expenses, with a floor of ฿0 when you are over budget.")
