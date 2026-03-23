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

def safe_ai_call(messages, system=None, max_tokens=600):
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
    messages = [format_message(m["role"], m["content"]) for m in chat_history]
    return safe_ai_call(messages, system="You are a senior DFM engineer.")


def get_ai_advice(geo, issues):
    prompt = f"Part: {geo}\nIssues: {issues}\nGive manufacturability advice with fixes."
    return safe_ai_call([format_message("user", prompt)])


def get_machining_guide(geo):
    prompt = f"Generate step-by-step CNC machining guide for part: {geo}"
    return safe_ai_call([format_message("user", prompt)], max_tokens=800)


def get_material_comparison(geo, materials):
    prompt = f"Compare these materials for part {geo}: {materials}"
    return safe_ai_call([format_message("user", prompt)], max_tokens=800)


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


def run_dfm_checks(geo, thresholds):
    issues = []
    min_dim = min(geo["length"], geo["width"], geo["height"])
    max_dim = max(geo["length"], geo["width"], geo["height"])

    if min_dim < thresholds["thin_wall"]:
        issues.append({"msg": "Thin wall detected", "penalty": 20})

    if max_dim / max(min_dim, 0.1) > thresholds["depth_ratio"]:
        issues.append({"msg": "Deep feature risk", "penalty": 20})

    if not geo["watertight"]:
        issues.append({"msg": "Mesh not watertight", "penalty": 10})

    return issues


def calculate_score(issues):
    return max(0, 100 - sum(i["penalty"] for i in issues))


# ==============================
# UI
# ==============================

st.title("⚙ DFM Pro")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Analyzer",
    "🔧 Machining Guide",
    "🧪 Materials",
    "💬 Chat",
    "⚙ Settings"
])

# ==============================
# SETTINGS TAB
# ==============================

with tab5:
    st.subheader("Thresholds")

    thin_wall = st.slider("Thin wall (mm)", 0.5, 5.0, 2.0)
    depth_ratio = st.slider("Depth ratio", 2, 15, 5)

    thresholds = {
        "thin_wall": thin_wall,
        "depth_ratio": depth_ratio
    }

# ==============================
# ANALYZER TAB
# ==============================

with tab1:
    file = st.file_uploader("Upload STL", type=["stl"])

    if file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        mesh = trimesh.load(path)
        os.unlink(path)

        geo = analyze_geometry(mesh)
        issues = run_dfm_checks(geo, thresholds)
        score = calculate_score(issues)

        st.session_state["geo"] = geo
        st.session_state["issues"] = issues

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Geometry")
            st.json(geo)

        with col2:
            st.metric("DFM Score", score)

        st.subheader("Issues")
        if not issues:
            st.success("No issues")
        else:
            for i in issues:
                st.warning(i["msg"])

        st.subheader("AI Advice")
        advice = get_ai_advice(geo, issues)
        st.info(advice)

# ==============================
# MACHINING GUIDE TAB
# ==============================

with tab2:
    if "geo" not in st.session_state:
        st.warning("Upload a file first")
    else:
        if st.button("Generate Guide"):
            guide = get_machining_guide(st.session_state["geo"])
            st.text_area("Guide", guide, height=400)

# ==============================
# MATERIAL TAB
# ==============================

with tab3:
    if "geo" not in st.session_state:
        st.warning("Upload a file first")
    else:
        materials = st.multiselect("Select materials", [
            "Aluminium 6061",
            "Steel",
            "Titanium",
            "Brass"
        ])

        if len(materials) >= 2:
            if st.button("Compare"):
                result = get_material_comparison(st.session_state["geo"], materials)
                st.text_area("Comparison", result, height=400)

# ==============================
# CHAT TAB
# ==============================

with tab4:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        st.write(f"**{msg['role']}**: {msg['content']}")

    user_input = st.chat_input("Ask...")

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        reply = get_chat_response(st.session_state.chat)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()
