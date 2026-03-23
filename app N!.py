import streamlit as st
import trimesh
import tempfile
import os
import anthropic
import numpy as np

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="DFM Pro", layout="wide")

API_KEY = st.secrets["ANTHROPIC_API_KEY"]
MODEL = "claude-3-5-sonnet-latest"

client = anthropic.Anthropic(api_key=API_KEY)

# ==============================
# AI ENGINE (FIXED)
# ==============================

def format_message(role, text):
    return {
        "role": role,
        "content": [{"type": "text", "text": str(text)}]
    }

def safe_ai_call(messages, system=None, max_tokens=500):
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠ AI Error: {str(e)}"


def get_chat_response(chat_history):
    messages = [
        format_message(m["role"], m["content"])
        for m in chat_history
        if m["role"] in ["user", "assistant"]
    ]

    if not messages:
        return "Ask something first."

    return safe_ai_call(
        messages,
        system="You are a senior CNC manufacturing engineer. Be concise and practical."
    )


def get_ai_advice(geo, issues):
    prompt = f"""
    Part: {geo}
    Issues: {issues}

    Give DFM advice:
    1. Manufacturability verdict
    2. Biggest issue
    3. Fix suggestions with numbers
    """

    return safe_ai_call([
        format_message("user", prompt)
    ])


# ==============================
# DFM CORE
# ==============================

def analyze_geometry(mesh):
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]

    return {
        "length": float(size[0]),
        "width": float(size[1]),
        "height": float(size[2]),
        "volume": float(mesh.volume) if mesh.is_volume else None,
        "faces": len(mesh.faces),
        "watertight": mesh.is_watertight
    }


def run_dfm_checks(geo):
    issues = []

    min_dim = min(geo["length"], geo["width"], geo["height"])
    max_dim = max(geo["length"], geo["width"], geo["height"])

    if min_dim < 1:
        issues.append({
            "severity": "critical",
            "msg": f"Too thin: {min_dim:.2f} mm",
            "penalty": 30
        })

    if max_dim / max(min_dim, 0.1) > 10:
        issues.append({
            "severity": "warning",
            "msg": "High aspect ratio",
            "penalty": 20
        })

    if not geo["watertight"]:
        issues.append({
            "severity": "warning",
            "msg": "Mesh not watertight",
            "penalty": 10
        })

    return issues


def calculate_score(issues):
    return max(0, 100 - sum(i["penalty"] for i in issues))


# ==============================
# UI
# ==============================

st.title("⚙ DFM Pro — AI Manufacturing Reviewer")

tab1, tab2 = st.tabs(["📁 Analyzer", "💬 Chat"])

# ==============================
# TAB 1 — ANALYZER
# ==============================

with tab1:
    file = st.file_uploader("Upload STL file", type=["stl"])

    if file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        mesh = trimesh.load(path)
        os.unlink(path)

        geo = analyze_geometry(mesh)
        issues = run_dfm_checks(geo)
        score = calculate_score(issues)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Geometry")
            st.json(geo)

        with col2:
            st.subheader("DFM Score")
            st.metric("Score", score)

        st.subheader("Issues")

        if not issues:
            st.success("No issues detected")
        else:
            for i in issues:
                st.warning(i["msg"])

        st.subheader("AI Engineering Advice")

        with st.spinner("Analyzing..."):
            advice = get_ai_advice(geo, issues)

        st.info(advice)


# ==============================
# TAB 2 — CHAT
# ==============================

with tab2:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        st.write(f"**{msg['role']}**: {msg['content']}")

    user_input = st.chat_input("Ask a manufacturing question...")

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})

        reply = get_chat_response(st.session_state.chat)

        st.session_state.chat.append({"role": "assistant", "content": reply})

        st.rerun()
