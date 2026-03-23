import streamlit as st
import trimesh
import tempfile
import os
import anthropic
import numpy as np
import json

st.set_page_config(
    page_title="DFM Pro",
    page_icon="⚙️",
    layout="wide"
)

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
MODEL = "claude-3-5-sonnet-latest"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0a0a0a; }
    .block-container { padding: 2rem 3rem; max-width: 1400px; }
    .hero-title { font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; font-weight: 600; color: #f0f0f0; letter-spacing: -1px; margin-bottom: 0.2rem; }
    .hero-sub { font-size: 0.95rem; color: #555; margin-bottom: 1.5rem; font-weight: 300; }
    .section-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #444; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.8rem; margin-top: 1.5rem; border-bottom: 1px solid #1a1a1a; padding-bottom: 0.4rem; }
    .metric-card { background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 1rem 1.2rem; text-align: center; }
    .metric-label { font-size: 0.65rem; color: #444; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem; }
    .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #e0e0e0; }
    .metric-unit { font-size: 0.7rem; color: #444; margin-top: 0.2rem; }
    .issue-card { border-radius: 6px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem; border-left: 3px solid; }
    .issue-critical { background: #1a0a0a; border-color: #e53e3e; color: #fc8181; }
    .issue-warning { background: #1a1400; border-color: #d69e2e; color: #f6e05e; }
    .issue-info { background: #0a1628; border-color: #3182ce; color: #90cdf4; }
    .issue-ok { background: #0a1a0f; border-color: #38a169; color: #68d391; }
    .issue-title { font-weight: 600; font-size: 0.85rem; margin-bottom: 0.15rem; }
    .issue-desc { font-size: 0.78rem; opacity: 0.8; }
    .ai-box { background: #0d1117; border: 1px solid #1e1e1e; border-left: 3px solid #3182ce; border-radius: 8px; padding: 1.2rem; font-size: 0.875rem; color: #c0c0c0; line-height: 1.7; }
    .ai-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #3182ce; letter-spacing: 0.1em; margin-bottom: 0.6rem; }
    .stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
    .stat-pill { background: #111; border: 1px solid #1e1e1e; border-radius: 20px; padding: 0.25rem 0.8rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #777; }
    .mat-card { background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem; }
    .mat-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; font-weight: 600; color: #e0e0e0; margin-bottom: 0.2rem; }
    .mat-reason { font-size: 0.78rem; color: #777; }
    .process-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; margin-right: 0.3rem; margin-top: 0.3rem; background: #1a2a1a; color: #68d391; border: 1px solid #2a4a2a; }
    .chat-user { display:flex; justify-content:flex-end; margin-bottom:0.8rem; }
    .chat-user-bubble { background:#1a2a3a; border:1px solid #1e3a5a; border-radius:8px 8px 2px 8px; padding:0.8rem 1rem; max-width:75%; font-size:0.875rem; color:#90cdf4; }
    .chat-ai { display:flex; justify-content:flex-start; margin-bottom:0.8rem; }
    .chat-ai-bubble { background:#111; border:1px solid #1e1e1e; border-left:3px solid #3182ce; border-radius:8px 8px 8px 2px; padding:0.8rem 1rem; max-width:75%; font-size:0.875rem; color:#c0c0c0; line-height:1.6; }
    .step-card { background:#111; border:1px solid #1e1e1e; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.6rem; border-left:3px solid #3182ce; }
    .step-num { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#3182ce; letter-spacing:0.1em; margin-bottom:0.3rem; }
    .step-title { font-size:0.9rem; font-weight:600; color:#e0e0e0; margin-bottom:0.3rem; }
    .step-detail { font-size:0.78rem; color:#888; line-height:1.6; }
    div[data-testid="stTabs"] button { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ---- AI FUNCTIONS ----

def safe_ai_call(messages, system=None, max_tokens=600):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        kwargs = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "messages": messages
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text
    except Exception as e:
        return f"AI Error: {str(e)}"


def get_ai_advice(geo, issues):
    issues_text = "\n".join([f"- {i['msg']} (penalty: {i['penalty']})" for i in issues]) if issues else "- No major issues."
    prompt = f"""You are a senior DFM engineer at a precision CNC machining company.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Faces: {geo['faces']:,}
Watertight: {geo['watertight']}

Issues:
{issues_text}

Write a concise assessment (max 200 words):
1. Overall manufacturability verdict
2. Most critical problem and real-world impact
3. Top 3 specific design changes with exact values
4. Recommended machining process

Be direct. Use mm values. No filler sentences."""
    return safe_ai_call([{"role": "user", "content": prompt}])


def get_machining_guide(geo, material):
    prompt = f"""You are a master CNC machinist with 25 years experience.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Material: {material}
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}

Write a complete step-by-step machining guide from raw stock to finished part.
For each step include: tool to use, cutting parameters (RPM, feed rate), what to watch out for.
Cover: raw stock selection, workholding, roughing, finishing, holes/features, inspection.
Max 600 words. Be very specific with tool sizes and parameters."""
    return safe_ai_call([{"role": "user", "content": prompt}], max_tokens=800)


def get_material_comparison(geo, materials_list):
    prompt = f"""You are a senior manufacturing engineer and materials specialist.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm

Compare these materials:
{chr(10).join([f"- {m}" for m in materials_list])}

For each material:
1. Machinability score (1-10)
2. Difficulty for this part
3. Key machining challenges
4. Surface finish achievable
5. Cost tier (Low/Medium/High/Very High)

End with a clear recommendation. Be specific with numbers."""
    return safe_ai_call([{"role": "user", "content": prompt}], max_tokens=800)


def get_chat_response(chat_history):
    messages = []
    for m in chat_history:
        if m["role"] in ["user", "assistant"]:
            messages.append({
                "role": m["role"],
                "content": str(m["content"])
            })
    if not messages:
        return "Please ask a question."
    system = """You are a senior DFM (Design for Manufacturing) engineer with 20 years experience in CNC machining, turning, EDM, and precision manufacturing.
Answer questions concisely and practically. Use specific numbers and values.
Use line breaks between points. No markdown headers, no bullet symbols, just clean text."""
    return safe_ai_call(messages, system=system, max_tokens=400)


# ---- GEOMETRY FUNCTIONS ----

def analyze_geometry(mesh):
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    volume = float(mesh.volume) if mesh.is_volume else None
    bbox_volume = float(size[0] * size[1] * size[2])
    fill_ratio = (volume / bbox_volume * 100) if volume and bbox_volume > 0 else None
    return {
        "length": float(size[0]),
        "width": float(size[1]),
        "height": float(size[2]),
        "volume": volume,
        "surface_area": float(mesh.area),
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "watertight": mesh.is_watertight,
        "fill_ratio": fill_ratio
    }


def run_dfm_checks(geo, thresholds):
    issues = []
    dims = sorted([geo["length"], geo["width"], geo["height"]])
    min_dim, mid_dim, max_dim = dims

    if min_dim < thresholds["critical_wall"]:
        issues.append({"severity": "critical", "msg": f"Extremely thin feature ({min_dim:.2f}mm) — impossible to machine", "penalty": 35, "tag": "thin_wall"})
    elif min_dim < thresholds["thin_wall"]:
        issues.append({"severity": "warning", "msg": f"Thin wall ({min_dim:.2f}mm) — risk of breakage", "penalty": 20, "tag": "thin_wall"})

    ratio = max_dim / max(min_dim, 0.01)
    if ratio > thresholds["depth_ratio"] * 2:
        issues.append({"severity": "critical", "msg": f"Extreme depth ratio {ratio:.1f}x — standard tooling cannot reach", "penalty": 35, "tag": "deep_feature"})
    elif ratio > thresholds["depth_ratio"]:
        issues.append({"severity": "warning", "msg": f"Deep feature ratio {ratio:.1f}x (limit: {thresholds['depth_ratio']}x)", "penalty": 20, "tag": "deep_feature"})

    if max_dim > thresholds["max_size"] * 2:
        issues.append({"severity": "critical", "msg": f"Exceeds machine envelope ({max_dim:.1f}mm)", "penalty": 30, "tag": "oversized"})
    elif max_dim > thresholds["max_size"]:
        issues.append({"severity": "warning", "msg": f"Large part ({max_dim:.1f}mm) — high machining cost", "penalty": 15, "tag": "oversized"})

    slender = max_dim / max(mid_dim, 0.01)
    if slender > 10:
        issues.append({"severity": "critical", "msg": f"Extremely slender {slender:.1f}:1 — severe deflection risk", "penalty": 30, "tag": "slender"})
    elif slender > 5:
        issues.append({"severity": "warning", "msg": f"Slender geometry {slender:.1f}:1 — may deflect during machining", "penalty": 15, "tag": "slender"})

    if not geo["watertight"]:
        issues.append({"severity": "warning", "msg": "Non-watertight mesh — fix before manufacturing", "penalty": 10, "tag": "mesh"})

    if geo["faces"] > 100000:
        issues.append({"severity": "info", "msg": f"High complexity ({geo['faces']:,} faces) — may need 5-axis", "penalty": 10, "tag": "complexity"})

    return issues


def calculate_score(issues):
    return max(0, min(100, 100 - sum(i["penalty"] for i in issues)))


def get_cost_tier(score):
    if score >= 85:
        return "LOW", "#38a169", "#0a1a0f"
    elif score >= 60:
        return "MEDIUM", "#d69e2e", "#1a1400"
    elif score >= 35:
        return "HIGH", "#e53e3e", "#1a0a0a"
    else:
        return "VERY HIGH", "#9b2c2c", "#150505"


def estimate_cost(geo, score, material="Aluminium 6061"):
    volume_cm3 = (geo["volume"] / 1000) if geo["volume"] else (geo["length"] * geo["width"] * geo["height"] / 1000 * 0.6)
    rates = {
        "Aluminium 6061": 350, "Mild Steel (1018)": 180,
        "Stainless Steel 304": 520, "Titanium Grade 5": 3500,
        "Brass C360": 650, "Free-machining Steel (12L14)": 220, "Tool Steel D2": 800,
    }
    rate = rates.get(material, 350)
    material_cost = volume_cm3 * rate * 0.00785
    machining_cost = volume_cm3 * 800 * (1.0 + (100 - score) / 100 * 2.5)
    setup_cost = 2500 if score > 70 else 5000 if score > 40 else 9000
    total_inr = material_cost + machining_cost + setup_cost
    return {
        "material_inr": round(material_cost),
        "machining_inr": round(machining_cost),
        "setup_inr": round(setup_cost),
        "total_inr": round(total_inr),
        "total_usd": round(total_inr / 83.5, 2)
    }


def get_problem_faces(mesh, issues):
    problem_faces = set()
    normals = mesh.face_normals
    issue_tags = [i.get("tag", "") for i in issues]
    if "slender" in issue_tags or "deep_feature" in issue_tags:
        centers = mesh.triangles_center
        main_axis = np.argmax(mesh.bounds[1] - mesh.bounds[0])
        axis_vals = centers[:, main_axis]
        low = np.percentile(axis_vals, 15)
        high = np.percentile(axis_vals, 85)
        for i, val in enumerate(axis_vals):
            if val < low or val > high:
                problem_faces.add(i)
    if "thin_wall" in issue_tags:
        thin_axis = np.argmin(mesh.bounds[1] - mesh.bounds[0])
        axis_vec = np.zeros(3)
        axis_vec[thin_axis] = 1.0
        dots = np.abs(np.dot(normals, axis_vec))
        for i, d in enumerate(dots):
            if d > 0.85:
                problem_faces.add(i)
    return list(problem_faces)


def mesh_to_threejs_json(mesh, problem_face_indices):
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    problem_set = set(problem_face_indices)
    positions = []
    colors = []
    for i, face in enumerate(faces):
        for vi in face:
            positions.extend(vertices[vi])
            if i in problem_set:
                colors.extend([0.9, 0.1, 0.1])
            else:
                colors.extend([0.45, 0.48, 0.52])
    return json.dumps({
        "positions": positions,
        "colors": colors,
        "problem_count": len(problem_set)
    })


def render_3d_viewer(mesh_json_str, score):
    score_color = "#38a169" if score >= 85 else "#d69e2e" if score >= 60 else "#e53e3e"
    return f"""<!DOCTYPE html>
<html><head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d0d0d; overflow:hidden; }}
  canvas {{ display:block; }}
  #info {{ position:absolute; top:12px; left:12px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#555; }}
  #legend {{ position:absolute; bottom:12px; left:12px; font-family:'IBM Plex Mono',monospace; font-size:10px; color:#555; display:flex; gap:16px; align-items:center; }}
  .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }}
  #controls {{ position:absolute; top:12px; right:12px; display:flex; flex-direction:column; gap:6px; }}
  button {{ background:#1a1a1a; border:1px solid #2a2a2a; color:#777; font-family:'IBM Plex Mono',monospace; font-size:10px; padding:5px 10px; border-radius:4px; cursor:pointer; }}
  button:hover {{ background:#222; color:#aaa; }}
  #badge {{ position:absolute; top:12px; left:50%; transform:translateX(-50%); font-family:'IBM Plex Mono',monospace; font-size:11px; color:{score_color}; background:#0d0d0d; border:1px solid {score_color}44; border-radius:4px; padding:4px 12px; }}
</style>
</head><body>
<canvas id="c"></canvas>
<div id="info">drag to rotate · scroll to zoom</div>
<div id="badge">Score: {score}/100</div>
<div id="legend">
  <span><span class="dot" style="background:#738085"></span>Normal</span>
  <span><span class="dot" style="background:#e53e3e"></span>Problem</span>
</div>
<div id="controls">
  <button onclick="rotX=0.3;rotY=0.5;zoom=1">Reset</button>
  <button onclick="wire=!wire;mesh.material=wire?wMat:mat">Wireframe</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const data = {mesh_json_str};
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas,antialias:true}});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth,window.innerHeight);
renderer.setClearColor(0x0d0d0d,1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45,window.innerWidth/window.innerHeight,0.01,10000);
const geo = new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(data.positions),3));
geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(data.colors),3));
geo.computeVertexNormals();
const mat = new THREE.MeshPhongMaterial({{vertexColors:true,shininess:40,specular:new THREE.Color(0x222222),side:THREE.DoubleSide}});
const wMat = new THREE.MeshBasicMaterial({{color:0x334455,wireframe:true}});
const mesh = new THREE.Mesh(geo,mat);
scene.add(mesh);
geo.computeBoundingBox();
const center = new THREE.Vector3();
geo.boundingBox.getCenter(center);
mesh.position.sub(center);
const sz = new THREE.Vector3();
geo.boundingBox.getSize(sz);
const maxDim = Math.max(sz.x,sz.y,sz.z);
camera.position.set(maxDim*1.2,maxDim*0.8,maxDim*1.5);
camera.lookAt(0,0,0);
scene.add(new THREE.AmbientLight(0xffffff,0.4));
const d1=new THREE.DirectionalLight(0xffffff,0.8); d1.position.set(1,2,3); scene.add(d1);
const d2=new THREE.DirectionalLight(0x8899bb,0.3); d2.position.set(-2,-1,-1); scene.add(d2);
const grid=new THREE.GridHelper(maxDim*3,20,0x1a1a1a,0x1a1a1a); grid.position.y=-maxDim*0.6; scene.add(grid);
let isDown=false,lastX=0,lastY=0,rotX=0.3,rotY=0.5,zoom=1,wire=false;
canvas.addEventListener('mousedown',e=>{{isDown=true;lastX=e.clientX;lastY=e.clientY;}});
canvas.addEventListener('mouseup',()=>isDown=false);
canvas.addEventListener('mousemove',e=>{{if(!isDown)return;rotY+=(e.clientX-lastX)*0.008;rotX+=(e.clientY-lastY)*0.008;lastX=e.clientX;lastY=e.clientY;}});
canvas.addEventListener('wheel',e=>{{zoom*=e.deltaY>0?1.1:0.9;zoom=Math.max(0.2,Math.min(5,zoom));}});
function animate(){{
  requestAnimationFrame(animate);
  mesh.rotation.x=rotX;mesh.rotation.y=rotY;
  camera.position.set(maxDim*1.2*zoom,maxDim*0.8*zoom,maxDim*1.5*zoom);
  camera.lookAt(0,0,0);renderer.render(scene,camera);
}}
animate();
window.addEventListener('resize',()=>{{camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);}});
</script></body></html>"""


# ---- UI ----

st.markdown('<div class="hero-title">⚙ DFM Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered Design for Manufacturing analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁  STL Analyzer",
    "🔧  Machining Guide",
    "🧪  Material Comparator",
    "💬  DFM Chat",
    "⚙  Settings"
])


# ======== SETTINGS ========
with tab5:
    st.markdown('<div class="section-title">Analysis Thresholds</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        thin_wall = st.slider("Thin wall warning (mm)", 0.5, 5.0, 2.0, 0.1)
        critical_wall = st.slider("Critical wall (mm)", 0.1, 2.0, 0.8, 0.1)
        depth_ratio = st.slider("Max depth/diameter ratio", 2, 15, 5)
    with col_s2:
        max_size = st.slider("Max part size warning (mm)", 100, 2000, 500, 50)
        default_material = st.selectbox("Default material", [
            "Aluminium 6061", "Mild Steel (1018)", "Stainless Steel 304",
            "Titanium Grade 5", "Brass C360", "Free-machining Steel (12L14)", "Tool Steel D2"
        ])

thresholds = {
    "thin_wall": thin_wall,
    "critical_wall": critical_wall,
    "depth_ratio": depth_ratio,
    "max_size": max_size
}


# ======== STL ANALYZER ========
with tab1:
    uploaded_file = st.file_uploader("", type=["stl"], label_visibility="collapsed")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Loading geometry..."):
                mesh = trimesh.load_mesh(tmp_path, file_type='stl')
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(mesh.dump())
                if mesh.is_empty:
                    st.error("Empty or invalid STL file.")
                    st.stop()
        except Exception as e:
            st.error(f"STL Load Error: {str(e)}")
            st.stop()
        finally:
            os.unlink(tmp_path)

        geo = analyze_geometry(mesh)
        issues = run_dfm_checks(geo, thresholds)
        score = calculate_score(issues)
        cost_tier, cost_color, cost_bg = get_cost_tier(score)
        cost_data = estimate_cost(geo, score, default_material)

        st.session_state["geo"] = geo
        st.session_state["issues"] = issues
        st.session_state["loaded"] = True

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown('<div class="section-title">3D Model Viewer</div>', unsafe_allow_html=True)
            problem_faces = get_problem_faces(mesh, issues)
            mesh_json = mesh_to_threejs_json(mesh, problem_faces)
            st.components.v1.html(render_3d_viewer(mesh_json, score), height=460, scrolling=False)

            if problem_faces:
                st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#e53e3e;margin-top:0.4rem">⚠ {len(problem_faces)} problem faces in red</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#38a169;margin-top:0.4rem">✓ No problem areas</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Geometry</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Length</div><div class="metric-value">{geo["length"]:.1f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Width</div><div class="metric-value">{geo["width"]:.1f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Height</div><div class="metric-value">{geo["height"]:.1f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
            with c4:
                vol = f'{geo["volume"]/1000:.1f}' if geo["volume"] else "N/A"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Volume</div><div class="metric-value">{vol}</div><div class="metric-unit">cm3</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            pills = [f"Faces: {geo['faces']:,}", f"Vertices: {geo['vertices']:,}", f"Surface: {geo['surface_area']:.0f} mm2", f"Watertight: {'Yes' if geo['watertight'] else 'No'}"]
            if geo["fill_ratio"]:
                pills.append(f"Fill: {geo['fill_ratio']:.1f}%")
            pills_html = "".join([f'<span class="stat-pill">{p}</span>' for p in pills])
            st.markdown(f'<div class="stat-row">{pills_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Cost Estimate</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total (INR)</div><div class="metric-value" style="font-size:1.4rem">Rs {cost_data["total_inr"]:,}</div><div class="metric-unit">{default_material}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total (USD)</div><div class="metric-value" style="font-size:1.4rem">${cost_data["total_usd"]:,}</div><div class="metric-unit">approx</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#555;margin-top:0.5rem">Material: Rs {cost_data["material_inr"]:,} · Machining: Rs {cost_data["machining_inr"]:,} · Setup: Rs {cost_data["setup_inr"]:,}</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="section-title">Manufacturability Score</div>', unsafe_allow_html=True)
            score_color = "#38a169" if score >= 85 else "#d69e2e" if score >= 60 else "#e53e3e"
            verdict = "Ready to manufacture" if score >= 85 else "Needs minor revisions" if score >= 60 else "Significant redesign needed" if score >= 35 else "Not manufacturable as-is"
            sc1, sc2 = st.columns([1, 3])
            with sc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Score</div><div class="metric-value" style="color:{score_color};font-size:2.5rem">{score}</div><div class="metric-unit">/ 100</div></div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'<div class="metric-card" style="text-align:left"><div style="color:{score_color};font-size:1rem;font-weight:600;margin-bottom:0.4rem">{verdict}</div><div style="color:#444;font-size:0.8rem;margin-bottom:0.6rem">{len(issues)} issue(s) detected</div><span style="background:{cost_bg};color:{cost_color};border:1px solid {cost_color}44;padding:0.2rem 0.7rem;border-radius:20px;font-family:IBM Plex Mono,monospace;font-size:0.75rem;font-weight:600">{cost_tier} COST</span></div>', unsafe_allow_html=True)
            st.progress(score / 100)

            st.markdown('<div class="section-title">DFM Issues</div>', unsafe_allow_html=True)
            if not issues:
                st.markdown('<div class="issue-card issue-ok"><div class="issue-title">✓ No issues detected</div><div class="issue-desc">Part meets manufacturability criteria.</div></div>', unsafe_allow_html=True)
            else:
                sev_order = {"critical": 0, "warning": 1, "info": 2}
                for issue in sorted(issues, key=lambda x: sev_order.get(x["severity"], 3)):
                    css = {"critical": "issue-critical", "warning": "issue-warning", "info": "issue-info"}.get(issue["severity"], "issue-info")
                    icon = {"critical": "✕", "warning": "⚠", "info": "ℹ"}.get(issue["severity"], "•")
                    st.markdown(f'<div class="issue-card {css}"><div class="issue-title">{icon} {issue["msg"]}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">AI Engineering Assessment</div>', unsafe_allow_html=True)
            with st.spinner("Generating assessment..."):
                advice = get_ai_advice(geo, issues)
            st.markdown(f'<div class="ai-box"><div class="ai-label">▸ CLAUDE AI — MANUFACTURING ENGINEER</div>{advice.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    else:
        st.session_state["loaded"] = False
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.2">⚙</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:#333">Upload an STL file to begin</div>
            <div style="font-size:0.8rem;color:#2a2a2a;margin-top:0.5rem">Supports binary and ASCII STL</div>
        </div>
        """, unsafe_allow_html=True)


# ======== MACHINING GUIDE ========
with tab2:
    st.markdown('<div class="section-title">Complete Machining Guide</div>', unsafe_allow_html=True)
    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first in STL Analyzer</div></div>', unsafe_allow_html=True)
    else:
        guide_mat = st.selectbox("Material", ["Aluminium 6061", "Mild Steel (1018)", "Stainless Steel 304", "Titanium Grade 5", "Brass C360", "Free-machining Steel (12L14)", "Tool Steel D2"], key="guide_mat")
        if st.button("Generate Machining Guide"):
            with st.spinner("Generating guide..."):
                guide = get_machining_guide(st.session_state["geo"], guide_mat)
            st.session_state["guide"] = guide

        if "guide" in st.session_state:
            lines = st.session_state["guide"].strip().split("\n")
            step_num = 0
            current = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                is_step = line[0].isdigit() and len(line) > 2 and line[1] in ".)"
                if is_step and current:
                    step_num += 1
                    st.markdown(f'<div class="step-card"><div class="step-num">STEP {step_num}</div><div class="step-title">{current[0]}</div><div class="step-detail">{" ".join(current[1:])}</div></div>', unsafe_allow_html=True)
                    current = [line]
                elif is_step:
                    current = [line]
                else:
                    current.append(line)
            if current:
                step_num += 1
                st.markdown(f'<div class="step-card"><div class="step-num">STEP {step_num}</div><div class="step-title">{current[0]}</div><div class="step-detail">{" ".join(current[1:])}</div></div>', unsafe_allow_html=True)


# ======== MATERIAL COMPARATOR ========
with tab3:
    st.markdown('<div class="section-title">Material Comparator</div>', unsafe_allow_html=True)
    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first in STL Analyzer</div></div>', unsafe_allow_html=True)
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Select materials to compare**")
            common = ["Aluminium 6061", "Aluminium 7075", "Mild Steel (1018)", "Stainless Steel 304", "Stainless Steel 316", "Titanium Grade 5", "Brass C360", "Copper", "Free-machining Steel (12L14)", "Tool Steel D2", "Inconel 718", "Delrin (POM)"]
            selected = st.multiselect("", common, default=["Aluminium 6061", "Mild Steel (1018)"])
        with col_m2:
            st.markdown("**Add custom materials**")
            c1 = st.text_input("Custom 1", placeholder="e.g. Bronze C932")
            c2 = st.text_input("Custom 2", placeholder="e.g. PEEK plastic")

        all_mats = selected + [m for m in [c1, c2] if m.strip()]

        if len(all_mats) < 2:
            st.markdown('<div class="issue-card issue-warning"><div class="issue-title">⚠ Select at least 2 materials</div></div>', unsafe_allow_html=True)
        else:
            if st.button("Compare Materials"):
                with st.spinner("Comparing materials..."):
                    result = get_material_comparison(st.session_state["geo"], all_mats)
                st.session_state["mat_result"] = result
                st.session_state["mat_list"] = all_mats

        if "mat_result" in st.session_state:
            st.markdown(f'<div class="ai-box"><div class="ai-label">▸ CLAUDE AI — MATERIALS ENGINEER</div>{st.session_state["mat_result"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Cost Comparison</div>', unsafe_allow_html=True)
            geo = st.session_state["geo"]
            issues = st.session_state["issues"]
            score = calculate_score(issues)
            cols = st.columns(min(len(st.session_state["mat_list"]), 4))
            for i, mat in enumerate(st.session_state["mat_list"][:4]):
                cost = estimate_cost(geo, score, mat)
                with cols[i]:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">{mat[:18]}</div><div class="metric-value" style="font-size:1.1rem">Rs {cost["total_inr"]:,}</div><div class="metric-unit">${cost["total_usd"]}</div></div>', unsafe_allow_html=True)


# ======== DFM CHAT ========
with tab4:
    st.markdown('<div class="section-title">DFM Engineering Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1.5rem">Ask any manufacturing question.</div>', unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai"><div class="chat-ai-bubble">{msg["content"].replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)

    if len(st.session_state.chat) == 0:
        st.markdown('<div style="color:#333;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem">Suggested questions</div>', unsafe_allow_html=True)
        suggestions = [
            'How deep is too deep for a 3/8" hole?',
            "Minimum wall thickness for aluminium CNC parts?",
            "When should I use EDM instead of milling?",
            "What tolerances can I expect from CNC milling?",
            "How do I reduce machining cost on a complex part?",
            "Best surface finish for sliding components?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    try:
                        st.session_state.chat.append({"role": "user", "content": s})
                        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                        response = client.messages.create(
                            model=MODEL,
                            max_tokens=400,
                            system="You are a senior DFM engineer. Answer concisely with specific numbers. Clean text, no bullets.",
                            messages=[{"role": "user", "content": s}]
                        )
                        st.session_state.chat.append({"role": "assistant", "content": response.content[0].text})
                    except Exception as e:
                        st.session_state.chat.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.rerun()

    user_input = st.chat_input("Ask a manufacturing question...")
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        reply = get_chat_response(st.session_state.chat)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()

    if len(st.session_state.chat) > 0:
        if st.button("Clear chat"):
            st.session_state.chat = []
            st.rerun()
