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
MODEL = "claude-haiku-4-5-20251001"

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
    .summary-box { background: #0d1117; border: 1px solid #1e1e1e; border-left: 3px solid #38a169; border-radius: 8px; padding: 1.2rem; font-size: 0.875rem; color: #c0c0c0; line-height: 1.7; margin-bottom: 1rem; }
    .feature-card { background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.4rem; display: flex; justify-content: space-between; align-items: center; }
    .feature-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #e0e0e0; }
    .feature-detail { font-size: 0.75rem; color: #666; }
    .step-card { background:#111; border:1px solid #1e1e1e; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.6rem; border-left:3px solid #3182ce; }
    .step-num { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#3182ce; letter-spacing:0.1em; margin-bottom:0.3rem; }
    .step-title { font-size:0.9rem; font-weight:600; color:#e0e0e0; margin-bottom:0.3rem; }
    .step-detail { font-size:0.78rem; color:#888; line-height:1.6; }
</style>
""", unsafe_allow_html=True)


# ==================== AI FUNCTIONS ====================

def safe_ai(messages, system=None, max_tokens=600):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        kwargs = {"model": MODEL, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text
    except Exception as e:
        return f"AI Error: {str(e)}"


def generate_part_summary(geo, features, issues):
    issue_text = ", ".join([i["msg"] for i in issues]) if issues else "none"
    feature_text = f"{len(features['holes'])} holes detected, " if features["holes"] else ""
    prompt = f"""You are a senior DFM engineer. A part has been uploaded for analysis.

Dimensions: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Faces: {geo['faces']:,}
Watertight: {geo['watertight']}
Detected features: {feature_text}aspect ratio {geo['aspect_ratio']:.1f}:1
Issues found: {issue_text}

Write a 3-sentence engineering summary of this part:
1. What kind of part this likely is and its general geometry
2. Key manufacturability characteristics
3. Primary recommendation

Be specific and direct. No generic statements."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=200)


def get_context_chat_response(chat_history, part_context):
    system = f"""You are a senior DFM (Design for Manufacturing) engineer with 20 years experience.

The engineer has uploaded a part with these exact specifications:
{part_context}

IMPORTANT RULES:
- Every answer must be specific to THIS part, not generic
- Reference actual dimensions, features, and issues from the part data above
- If asked about material, recommend based on this part's geometry
- If asked about tools, recommend based on actual dimensions
- If asked about cost, reference the actual volume and complexity
- Never give generic textbook answers
- Use mm values from the actual part
- Keep answers concise and practical"""

    messages = []
    for m in chat_history:
        if m["role"] in ["user", "assistant"]:
            messages.append({"role": m["role"], "content": str(m["content"])})

    if not messages:
        return "Please ask a question about your part."

    return safe_ai(messages, system=system, max_tokens=500)


def get_ai_advice(geo, features, issues):
    issue_text = "\n".join([f"- {i['msg']}" for i in issues]) if issues else "- No major issues"
    holes_text = f"\nDetected holes: {len(features['holes'])} cylindrical features" if features["holes"] else ""
    prompt = f"""You are a senior DFM engineer.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Aspect ratio: {geo['aspect_ratio']:.1f}:1
Watertight: {geo['watertight']}{holes_text}

Issues:
{issue_text}

Write concise assessment (max 180 words):
1. Manufacturability verdict
2. Most critical issue and impact
3. Top 3 specific fixes with exact values
4. Recommended process

Direct, specific, use mm values."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=400)


def get_machining_guide(geo, material, features):
    holes_text = f"Detected {len(features['holes'])} holes" if features["holes"] else "No holes detected"
    prompt = f"""You are a master CNC machinist with 25 years experience.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Material: {material}
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Features: {holes_text}, aspect ratio {geo['aspect_ratio']:.1f}:1

Write complete step-by-step machining guide. For each step include exact tool, RPM, feed rate, what to watch for.
Cover: raw stock, workholding, roughing, finishing, holes, inspection.
Max 500 words. Be very specific."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=700)


def get_material_comparison(geo, materials_list, issues):
    issue_tags = [i.get("tag", "") for i in issues]
    prompt = f"""You are a senior materials and manufacturing engineer.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Issues: {', '.join(issue_tags) if issue_tags else 'none'}

Compare these materials for this specific part:
{chr(10).join([f"- {m}" for m in materials_list])}

For each material:
1. Machinability score (1-10)
2. Difficulty for THIS specific geometry
3. Key challenges
4. Cost tier (Low/Medium/High/Very High)

End with clear recommendation. Be specific."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=700)


# ==================== GEOMETRY FUNCTIONS ====================

def load_mesh(uploaded_file, file_ext):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if file_ext == ".stl":
            mesh = trimesh.load_mesh(tmp_path, file_type='stl')
        else:
            mesh = trimesh.load(tmp_path)
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if geometries:
                mesh = trimesh.util.concatenate(geometries)
            else:
                return None
        if mesh.is_empty:
            return None
        return mesh
    except Exception as e:
        st.error(f"Load error: {str(e)}")
        return None
    finally:
        os.unlink(tmp_path)


def analyze_geometry(mesh):
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    length, width, height = float(size[0]), float(size[1]), float(size[2])
    volume = float(mesh.volume) if mesh.is_volume else None
    dims = sorted([length, width, height])
    aspect_ratio = dims[2] / max(dims[1], 0.01)
    bbox_volume = length * width * height
    fill_ratio = (volume / bbox_volume * 100) if volume and bbox_volume > 0 else None
    return {
        "length": length, "width": width, "height": height,
        "volume": volume, "surface_area": float(mesh.area),
        "faces": len(mesh.faces), "vertices": len(mesh.vertices),
        "watertight": mesh.is_watertight,
        "fill_ratio": fill_ratio, "aspect_ratio": aspect_ratio,
        "bbox_volume": bbox_volume
    }


def detect_features(mesh):
    features = {"holes": [], "thin_regions": [], "complex_surfaces": False}

    try:
        normals = mesh.face_normals
        face_areas = mesh.area_faces
        total_faces = len(normals)

        # Detect cylindrical features (potential holes)
        # Group faces by normal direction clustering
        z_facing = np.abs(normals[:, 2])
        horizontal_mask = z_facing < 0.3
        horizontal_faces = np.where(horizontal_mask)[0]

        if len(horizontal_faces) > 20:
            centers = mesh.triangles_center[horizontal_faces]
            # Simple clustering by XY position to find circular patterns
            unique_regions = []
            used = set()
            for i, fc in enumerate(centers):
                if i in used:
                    continue
                nearby = []
                for j, fc2 in enumerate(centers):
                    if j in used:
                        continue
                    dist = np.sqrt((fc[0]-fc2[0])**2 + (fc[1]-fc2[1])**2)
                    if dist < 15:
                        nearby.append(j)
                if len(nearby) > 8:
                    region_centers = centers[nearby]
                    cx = float(np.mean(region_centers[:, 0]))
                    cy = float(np.mean(region_centers[:, 1]))
                    cz = float(np.mean(region_centers[:, 2]))
                    radii = [np.sqrt((c[0]-cx)**2 + (c[1]-cy)**2) for c in region_centers]
                    avg_r = float(np.mean(radii))
                    if avg_r > 1.0:
                        unique_regions.append({
                            "center": (cx, cy, cz),
                            "diameter": avg_r * 2,
                            "face_count": len(nearby)
                        })
                    for j in nearby:
                        used.add(j)

            features["holes"] = unique_regions[:6]

        if geo["faces"] > 50000:
            features["complex_surfaces"] = True

    except Exception:
        pass

    return features


def run_dfm_checks(geo, thresholds):
    issues = []
    dims = sorted([geo["length"], geo["width"], geo["height"]])
    min_dim, mid_dim, max_dim = dims

    if min_dim < thresholds["critical_wall"]:
        issues.append({"severity": "critical", "msg": f"Extremely thin feature {min_dim:.2f}mm — impossible to machine", "penalty": 35, "tag": "thin_wall"})
    elif min_dim < thresholds["thin_wall"]:
        issues.append({"severity": "warning", "msg": f"Thin wall {min_dim:.2f}mm — risk of breakage", "penalty": 20, "tag": "thin_wall"})

    ratio = max_dim / max(min_dim, 0.01)
    if ratio > thresholds["depth_ratio"] * 2:
        issues.append({"severity": "critical", "msg": f"Extreme depth ratio {ratio:.1f}x — standard tooling cannot reach", "penalty": 35, "tag": "deep_feature"})
    elif ratio > thresholds["depth_ratio"]:
        issues.append({"severity": "warning", "msg": f"Deep feature ratio {ratio:.1f}x (limit: {thresholds['depth_ratio']}x)", "penalty": 20, "tag": "deep_feature"})

    if max_dim > thresholds["max_size"] * 2:
        issues.append({"severity": "critical", "msg": f"Exceeds machine envelope {max_dim:.1f}mm", "penalty": 30, "tag": "oversized"})
    elif max_dim > thresholds["max_size"]:
        issues.append({"severity": "warning", "msg": f"Large part {max_dim:.1f}mm — high machining cost", "penalty": 15, "tag": "oversized"})

    slender = max_dim / max(mid_dim, 0.01)
    if slender > 10:
        issues.append({"severity": "critical", "msg": f"Extremely slender {slender:.1f}:1 — severe deflection risk", "penalty": 30, "tag": "slender"})
    elif slender > 5:
        issues.append({"severity": "warning", "msg": f"Slender geometry {slender:.1f}:1 — may deflect during machining", "penalty": 15, "tag": "slender"})

    if not geo["watertight"]:
        issues.append({"severity": "warning", "msg": "Non-watertight mesh — fix before manufacturing", "penalty": 10, "tag": "mesh"})
    if geo["faces"] > 100000:
        issues.append({"severity": "info", "msg": f"High complexity {geo['faces']:,} faces — may need 5-axis", "penalty": 10, "tag": "complexity"})
    if geo["fill_ratio"] and geo["fill_ratio"] < 20:
        issues.append({"severity": "info", "msg": f"Low fill ratio {geo['fill_ratio']:.1f}% — verify wall thicknesses", "penalty": 5, "tag": "hollow"})

    return issues


def calculate_score(issues):
    return max(0, min(100, 100 - sum(i["penalty"] for i in issues)))


def get_cost_tier(score):
    if score >= 85: return "LOW", "#38a169", "#0a1a0f"
    elif score >= 60: return "MEDIUM", "#d69e2e", "#1a1400"
    elif score >= 35: return "HIGH", "#e53e3e", "#1a0a0a"
    else: return "VERY HIGH", "#9b2c2c", "#150505"


def estimate_cost(geo, score, material="Aluminium 6061"):
    volume_cm3 = (geo["volume"] / 1000) if geo["volume"] else (geo["bbox_volume"] / 1000 * 0.6)
    rates = {
        "Aluminium 6061": 350, "Mild Steel (1018)": 180, "Stainless Steel 304": 520,
        "Titanium Grade 5": 3500, "Brass C360": 650,
        "Free-machining Steel (12L14)": 220, "Tool Steel D2": 800,
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
    positions, colors = [], []
    for i, face in enumerate(faces):
        for vi in face:
            positions.extend(vertices[vi])
            colors.extend([0.9, 0.1, 0.1] if i in problem_set else [0.45, 0.48, 0.52])
    return json.dumps({"positions": positions, "colors": colors, "problem_count": len(problem_set)})


def render_3d_viewer(mesh_json_str, score):
    score_color = "#38a169" if score >= 85 else "#d69e2e" if score >= 60 else "#e53e3e"
    return f"""<!DOCTYPE html><html><head>
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
  #badge {{ position:absolute; top:12px; left:50%; transform:translateX(-50%); font-family:'IBM Plex Mono',monospace; font-size:11px; color:{score_color}; background:#0d0d0d; border:1px solid {score_color}44; border-radius:4px; padding:4px 12px; white-space:nowrap; }}
</style></head><body>
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


def build_part_context(geo, features, issues, score, cost_data, material):
    issue_list = "\n".join([f"  - [{i['severity'].upper()}] {i['msg']}" for i in issues]) if issues else "  - None detected"
    holes_text = f"\n  - {len(features['holes'])} cylindrical features detected" if features["holes"] else "\n  - No holes detected"
    return f"""PART SPECIFICATIONS:
Dimensions: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Surface area: {geo['surface_area']:.1f} mm2
Aspect ratio: {geo['aspect_ratio']:.1f}:1
Faces: {geo['faces']:,}
Watertight: {geo['watertight']}
Fill ratio: {f"{geo['fill_ratio']:.1f}%" if geo['fill_ratio'] else 'N/A'}

DETECTED FEATURES:{holes_text}
  - Complex surfaces: {features['complex_surfaces']}

DFM ISSUES:
{issue_list}

MANUFACTURABILITY SCORE: {score}/100
COST ESTIMATE ({material}): Rs {cost_data['total_inr']:,} (${cost_data['total_usd']})
  - Material: Rs {cost_data['material_inr']:,}
  - Machining: Rs {cost_data['machining_inr']:,}
  - Setup: Rs {cost_data['setup_inr']:,}"""


# ==================== UI ====================

st.markdown('<div class="hero-title">⚙ DFM Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered Design for Manufacturing analysis — Phase 1</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁  STL / STEP Analyzer",
    "🔧  Machining Guide",
    "🧪  Material Comparator",
    "💬  Part Chat",
    "⚙  Settings"
])


# ==================== SETTINGS ====================
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
    "thin_wall": thin_wall, "critical_wall": critical_wall,
    "depth_ratio": depth_ratio, "max_size": max_size
}


# ==================== STL / STEP ANALYZER ====================
with tab1:
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1rem">Supports STL and STEP (.stp / .step) files</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["stl", "step", "stp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".stl"):
            file_ext = ".stl"
        elif file_name.endswith(".step"):
            file_ext = ".step"
        else:
            file_ext = ".stp"

        with st.spinner("Loading geometry..."):
            mesh = load_mesh(uploaded_file, file_ext)

        if mesh is None:
            st.error("Could not load file. Please check it is a valid STL or STEP file.")
            st.stop()

        geo = analyze_geometry(mesh)
        features = detect_features(mesh)
        issues = run_dfm_checks(geo, thresholds)
        score = calculate_score(issues)
        cost_tier, cost_color, cost_bg = get_cost_tier(score)
        cost_data = estimate_cost(geo, score, default_material)
        part_context = build_part_context(geo, features, issues, score, cost_data, default_material)

        st.session_state["geo"] = geo
        st.session_state["features"] = features
        st.session_state["issues"] = issues
        st.session_state["score"] = score
        st.session_state["part_context"] = part_context
        st.session_state["loaded"] = True
        st.session_state["filename"] = uploaded_file.name
        st.session_state["part_chat"] = []

        # Auto part summary
        st.markdown('<div class="section-title">AI Part Summary</div>', unsafe_allow_html=True)
        with st.spinner("Generating part summary..."):
            summary = generate_part_summary(geo, features, issues)
        st.markdown(f'<div class="summary-box"><div class="ai-label">▸ AUTO ANALYSIS — {uploaded_file.name.upper()}</div>{summary}</div>', unsafe_allow_html=True)

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown('<div class="section-title">3D Model Viewer</div>', unsafe_allow_html=True)
            problem_faces = get_problem_faces(mesh, issues)
            mesh_json = mesh_to_threejs_json(mesh, problem_faces)
            st.components.v1.html(render_3d_viewer(mesh_json, score), height=460, scrolling=False)

            if problem_faces:
                st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#e53e3e;margin-top:0.4rem">⚠ {len(problem_faces)} problem faces highlighted in red</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#38a169;margin-top:0.4rem">✓ No problem areas detected</div>', unsafe_allow_html=True)

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
            pills = [f"Faces: {geo['faces']:,}", f"Vertices: {geo['vertices']:,}", f"Surface: {geo['surface_area']:.0f} mm2", f"Watertight: {'Yes' if geo['watertight'] else 'No'}", f"Aspect ratio: {geo['aspect_ratio']:.1f}:1"]
            if geo["fill_ratio"]:
                pills.append(f"Fill: {geo['fill_ratio']:.1f}%")
            pills_html = "".join([f'<span class="stat-pill">{p}</span>' for p in pills])
            st.markdown(f'<div class="stat-row">{pills_html}</div>', unsafe_allow_html=True)

            # Detected features
            if features["holes"]:
                st.markdown('<div class="section-title">Detected Features</div>', unsafe_allow_html=True)
                for i, hole in enumerate(features["holes"]):
                    st.markdown(f'<div class="feature-card"><span class="feature-name">Cylindrical feature {i+1}</span><span class="feature-detail">dia ≈ {hole["diameter"]:.1f}mm</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Cost Estimate</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total (INR)</div><div class="metric-value" style="font-size:1.3rem">Rs {cost_data["total_inr"]:,}</div><div class="metric-unit">{default_material}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total (USD)</div><div class="metric-value" style="font-size:1.3rem">${cost_data["total_usd"]}</div><div class="metric-unit">approx</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#555;margin-top:0.5rem">Material: Rs {cost_data["material_inr"]:,} · Machining: Rs {cost_data["machining_inr"]:,} · Setup: Rs {cost_data["setup_inr"]:,}</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="section-title">Manufacturability Score</div>', unsafe_allow_html=True)
            score_color = "#38a169" if score >= 85 else "#d69e2e" if score >= 60 else "#e53e3e"
            verdict = "Ready to manufacture" if score >= 85 else "Needs minor revisions" if score >= 60 else "Significant redesign needed" if score >= 35 else "Not manufacturable as-is"
            sc1, sc2 = st.columns([1, 3])
            with sc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Score</div><div class="metric-value" style="color:{score_color};font-size:2.5rem">{score}</div><div class="metric-unit">/ 100</div></div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'<div class="metric-card" style="text-align:left"><div style="color:{score_color};font-size:1rem;font-weight:600;margin-bottom:0.4rem">{verdict}</div><div style="color:#444;font-size:0.8rem;margin-bottom:0.6rem">{len(issues)} issue(s) detected · {file_ext.upper()} file</div><span style="background:{cost_bg};color:{cost_color};border:1px solid {cost_color}44;padding:0.2rem 0.7rem;border-radius:20px;font-family:IBM Plex Mono,monospace;font-size:0.75rem;font-weight:600">{cost_tier} COST</span></div>', unsafe_allow_html=True)
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
                advice = get_ai_advice(geo, features, issues)
            st.markdown(f'<div class="ai-box"><div class="ai-label">▸ CLAUDE AI — MANUFACTURING ENGINEER</div>{advice.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    else:
        st.session_state["loaded"] = False
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.2">⚙</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:#333">Upload STL or STEP file to begin</div>
            <div style="font-size:0.8rem;color:#2a2a2a;margin-top:0.5rem">Supports .stl · .step · .stp formats</div>
        </div>
        """, unsafe_allow_html=True)


# ==================== MACHINING GUIDE ====================
with tab2:
    st.markdown('<div class="section-title">Complete Machining Guide</div>', unsafe_allow_html=True)
    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first in STL / STEP Analyzer</div></div>', unsafe_allow_html=True)
    else:
        guide_mat = st.selectbox("Material to machine in", [
            "Aluminium 6061", "Mild Steel (1018)", "Stainless Steel 304",
            "Titanium Grade 5", "Brass C360", "Free-machining Steel (12L14)", "Tool Steel D2"
        ], key="guide_mat")

        if st.button("Generate Machining Guide"):
            with st.spinner("Generating step-by-step guide..."):
                guide = get_machining_guide(st.session_state["geo"], guide_mat, st.session_state["features"])
            st.session_state["guide"] = guide

        if "guide" in st.session_state:
            lines = st.session_state["guide"].strip().split("\n")
            step_num = 0
            current = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                is_step = len(line) > 2 and line[0].isdigit() and line[1] in ".)"
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


# ==================== MATERIAL COMPARATOR ====================
with tab3:
    st.markdown('<div class="section-title">Material Comparator</div>', unsafe_allow_html=True)
    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first in STL / STEP Analyzer</div></div>', unsafe_allow_html=True)
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Select materials**")
            common = ["Aluminium 6061", "Aluminium 7075", "Mild Steel (1018)", "Stainless Steel 304",
                      "Stainless Steel 316", "Titanium Grade 5", "Brass C360", "Copper",
                      "Free-machining Steel (12L14)", "Tool Steel D2", "Inconel 718", "Delrin (POM)"]
            selected = st.multiselect("", common, default=["Aluminium 6061", "Mild Steel (1018)"])
        with col_m2:
            st.markdown("**Add custom materials**")
            c1 = st.text_input("Custom 1", placeholder="e.g. Bronze C932")
            c2 = st.text_input("Custom 2", placeholder="e.g. PEEK plastic")

        all_mats = selected + [m for m in [c1, c2] if m.strip()]

        if len(all_mats) < 2:
            st.markdown('<div class="issue-card issue-warning"><div class="issue-title">⚠ Select at least 2 materials to compare</div></div>', unsafe_allow_html=True)
        else:
            if st.button("Compare Materials"):
                with st.spinner("Comparing materials..."):
                    result = get_material_comparison(st.session_state["geo"], all_mats, st.session_state["issues"])
                st.session_state["mat_result"] = result
                st.session_state["mat_list"] = all_mats

        if "mat_result" in st.session_state:
            st.markdown(f'<div class="ai-box"><div class="ai-label">▸ CLAUDE AI — MATERIALS ENGINEER</div>{st.session_state["mat_result"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Cost Comparison</div>', unsafe_allow_html=True)
            geo = st.session_state["geo"]
            score = st.session_state["score"]
            cols = st.columns(min(len(st.session_state["mat_list"]), 4))
            for i, mat in enumerate(st.session_state["mat_list"][:4]):
                cost = estimate_cost(geo, score, mat)
                with cols[i]:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">{mat[:16]}</div><div class="metric-value" style="font-size:1.1rem">Rs {cost["total_inr"]:,}</div><div class="metric-unit">${cost["total_usd"]}</div></div>', unsafe_allow_html=True)


# ==================== PART CHAT ====================
with tab4:
    st.markdown('<div class="section-title">Part-Aware Engineering Chat</div>', unsafe_allow_html=True)

    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first — the chat will know your specific part</div></div>', unsafe_allow_html=True)
    else:
        geo = st.session_state["geo"]
        filename = st.session_state.get("filename", "your part")

        st.markdown(f'<div style="background:#0a1a0f;border:1px solid #1a3a1a;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#68d391">✓ AI knows your part: {filename} — {geo["length"]:.0f} x {geo["width"]:.0f} x {geo["height"]:.0f}mm — Score {st.session_state["score"]}/100</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1rem">Ask anything about your specific part — materials, tools, process, cost, fixes. AI answers based on your actual geometry.</div>', unsafe_allow_html=True)

        if "part_chat" not in st.session_state:
            st.session_state.part_chat = []

        for msg in st.session_state.part_chat:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai"><div class="chat-ai-bubble">{msg["content"].replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)

        if len(st.session_state.part_chat) == 0:
            st.markdown('<div style="color:#333;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem">Suggested questions about your part</div>', unsafe_allow_html=True)
            suggestions = [
                "What material should I use for this part?",
                "What drill size should I use?",
                "How can I reduce the cost of this part?",
                "Is this part suitable for CNC milling?",
                "What is the biggest manufacturing risk here?",
                "How can I make this easier to machine?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(s, key=f"psug_{i}", use_container_width=True):
                        try:
                            st.session_state.part_chat.append({"role": "user", "content": s})
                            reply = get_context_chat_response(
                                st.session_state.part_chat,
                                st.session_state["part_context"]
                            )
                            st.session_state.part_chat.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.session_state.part_chat.append({"role": "assistant", "content": f"Error: {str(e)}"})
                        st.rerun()

        user_input = st.chat_input("Ask about your specific part...")
        if user_input:
            st.session_state.part_chat.append({"role": "user", "content": user_input})
            reply = get_context_chat_response(
                st.session_state.part_chat,
                st.session_state["part_context"]
            )
            st.session_state.part_chat.append({"role": "assistant", "content": reply})
            st.rerun()

        if len(st.session_state.part_chat) > 0:
            if st.button("Clear chat", key="clear_part_chat"):
                st.session_state.part_chat = []
                st.rerun()
