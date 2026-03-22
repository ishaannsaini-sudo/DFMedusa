import streamlit as st
import trimesh
import tempfile
import os
import anthropic
import numpy as np
import json
from fpdf import FPDF
import io

st.set_page_config(
    page_title="DFM Pro — AI Manufacturing Reviewer",
    page_icon="⚙️",
    layout="wide"
)

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

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
</style>
""", unsafe_allow_html=True)


# ---- CORE FUNCTIONS ----

def analyze_geometry(mesh):
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    length, width, height = size
    volume = float(mesh.volume) if mesh.is_volume else None
    surface_area = float(mesh.area)
    face_count = len(mesh.faces)
    vertex_count = len(mesh.vertices)
    is_watertight = mesh.is_watertight
    bbox_volume = length * width * height
    fill_ratio = (volume / bbox_volume * 100) if volume and bbox_volume > 0 else None
    return {
        "length": length, "width": width, "height": height,
        "volume": volume, "surface_area": surface_area,
        "face_count": face_count, "vertex_count": vertex_count,
        "is_watertight": is_watertight, "fill_ratio": fill_ratio,
        "bbox_volume": bbox_volume
    }


def estimate_wall_thickness(mesh):
    try:
        samples, face_indices = trimesh.sample.sample_surface(mesh, 200)
        normals = mesh.face_normals[face_indices]
        origins = samples + normals * 0.01
        directions = -normals
        hits = mesh.ray.intersects_location(ray_origins=origins, ray_directions=directions)
        thicknesses = []
        if len(hits[0]) > 0:
            for i, loc in enumerate(hits[0]):
                dist = np.linalg.norm(loc - origins[hits[1][i]])
                if 0.01 < dist < 500:
                    thicknesses.append(dist)
        if thicknesses:
            return {"min": float(np.min(thicknesses)), "max": float(np.max(thicknesses)), "mean": float(np.mean(thicknesses))}
    except Exception:
        pass
    return None


def detect_undercuts(mesh):
    axes = {
        "Top (Z+)": np.array([0, 0, 1]),
        "Bottom (Z-)": np.array([0, 0, -1]),
        "Front (Y+)": np.array([0, 1, 0]),
        "Back (Y-)": np.array([0, -1, 0]),
        "Left (X+)": np.array([1, 0, 0]),
        "Right (X-)": np.array([-1, 0, 0]),
    }
    normals = mesh.face_normals
    total_faces = len(normals)
    results = {}
    for axis_name, axis_dir in axes.items():
        dots = np.dot(normals, axis_dir)
        undercut_faces = np.sum(dots < -0.1)
        results[axis_name] = {
            "undercut_faces": int(undercut_faces),
            "undercut_pct": float(undercut_faces / total_faces * 100) if total_faces > 0 else 0
        }
    best_axis = min(results, key=lambda x: results[x]["undercut_pct"])
    return {
        "axes": results,
        "best_axis": best_axis,
        "best_pct": results[best_axis]["undercut_pct"],
        "needs_multi_axis": results[best_axis]["undercut_pct"] > 15
    }


def run_dfm_checks(geo, thresholds):
    issues = []
    dims = sorted([geo["length"], geo["width"], geo["height"]])
    min_dim, mid_dim, max_dim = dims

    if min_dim < thresholds["critical_wall"]:
        issues.append({"severity": "critical", "title": "Extremely thin feature", "desc": f"Minimum dimension {min_dim:.2f}mm is below {thresholds['critical_wall']}mm — likely impossible to machine.", "tag": "thin_wall", "penalty": 35})
    elif min_dim < thresholds["thin_wall"]:
        issues.append({"severity": "warning", "title": "Thin wall detected", "desc": f"Minimum dimension {min_dim:.2f}mm is below {thresholds['thin_wall']}mm — risk of breakage.", "tag": "thin_wall", "penalty": 20})

    ratio = max_dim / min_dim
    if ratio > thresholds["depth_ratio"] * 2:
        issues.append({"severity": "critical", "title": "Extreme depth ratio", "desc": f"Depth/diameter ratio is {ratio:.1f}x — standard tooling cannot reach.", "tag": "deep_feature", "penalty": 35})
    elif ratio > thresholds["depth_ratio"]:
        issues.append({"severity": "warning", "title": "Deep feature detected", "desc": f"Depth/diameter ratio is {ratio:.1f}x (your limit: {thresholds['depth_ratio']}x).", "tag": "deep_feature", "penalty": 20})

    if max_dim > thresholds["max_size"] * 2:
        issues.append({"severity": "critical", "title": "Exceeds machine envelope", "desc": f"Max dimension {max_dim:.1f}mm exceeds typical CNC envelope.", "tag": "oversized", "penalty": 30})
    elif max_dim > thresholds["max_size"]:
        issues.append({"severity": "warning", "title": "Large part", "desc": f"Max dimension {max_dim:.1f}mm exceeds your limit of {thresholds['max_size']}mm.", "tag": "oversized", "penalty": 15})

    slender_ratio = max_dim / mid_dim if mid_dim > 0 else 0
    if slender_ratio > 10:
        issues.append({"severity": "critical", "title": "Extremely slender part", "desc": f"Aspect ratio {slender_ratio:.1f}:1 — severe deflection risk.", "tag": "slender", "penalty": 30})
    elif slender_ratio > 5:
        issues.append({"severity": "warning", "title": "Slender geometry", "desc": f"Aspect ratio {slender_ratio:.1f}:1 — may deflect during machining.", "tag": "slender", "penalty": 15})

    if not geo["is_watertight"]:
        issues.append({"severity": "warning", "title": "Non-watertight mesh", "desc": "Mesh has open edges — fix geometry before manufacturing.", "tag": "mesh_quality", "penalty": 10})
    if geo["face_count"] > 100000:
        issues.append({"severity": "info", "title": "High geometry complexity", "desc": f"{geo['face_count']:,} faces — may require 5-axis machining.", "tag": "complexity", "penalty": 10})
    if geo["fill_ratio"] and geo["fill_ratio"] < 20:
        issues.append({"severity": "info", "title": "Low material fill ratio", "desc": f"Part volume is only {geo['fill_ratio']:.1f}% of bounding box.", "tag": "hollow", "penalty": 5})
    if max_dim < 5:
        issues.append({"severity": "warning", "title": "Micro-scale part", "desc": f"Max dimension only {max_dim:.2f}mm — requires precision machining.", "tag": "micro", "penalty": 15})
    return issues


def calculate_score(issues):
    return max(0, min(100, 100 - sum(i.get("penalty", 0) for i in issues)))


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
    volume_cm3 = (geo["volume"] / 1000) if geo["volume"] else (geo["bbox_volume"] / 1000 * 0.6)
    material_rates_inr = {
        "Aluminium 6061": 350,
        "Mild Steel (1018)": 180,
        "Stainless Steel 304": 520,
        "Titanium Grade 5": 3500,
        "Brass C360": 650,
        "Free-machining Steel (12L14)": 220,
        "Tool Steel D2": 800,
    }
    rate_inr = material_rates_inr.get(material, 350)
    material_cost_inr = volume_cm3 * rate_inr * 0.00785

    complexity_multiplier = 1.0 + (100 - score) / 100 * 2.5
    machining_cost_inr = volume_cm3 * 800 * complexity_multiplier
    setup_cost_inr = 2500 if score > 70 else 5000 if score > 40 else 9000
    total_inr = material_cost_inr + machining_cost_inr + setup_cost_inr
    total_usd = total_inr / 83.5

    return {
        "material_inr": round(material_cost_inr),
        "machining_inr": round(machining_cost_inr),
        "setup_inr": round(setup_cost_inr),
        "total_inr": round(total_inr),
        "total_usd": round(total_usd, 2),
        "volume_cm3": round(volume_cm3, 2)
    }


def recommend_materials_and_process(geo, issues):
    dims = sorted([geo["length"], geo["width"], geo["height"]])
    min_dim, mid_dim, max_dim = dims
    issue_tags = [i["tag"] for i in issues]
    materials = []
    processes = []
    if "thin_wall" in issue_tags:
        materials.append({"name": "Aluminium 6061", "reason": "High strength-to-weight, machines well at thin sections"})
        materials.append({"name": "Titanium Grade 5", "reason": "Excellent for thin high-stress features"})
    else:
        materials.append({"name": "Aluminium 6061", "reason": "Best general-purpose — easy to machine, low cost, good strength"})
        materials.append({"name": "Mild Steel (1018)", "reason": "Good for structural parts, widely available"})
    if "deep_feature" in issue_tags:
        materials.append({"name": "Free-machining Steel (12L14)", "reason": "Better chip breaking for deep features"})
    if min_dim < 2:
        materials.append({"name": "Brass C360", "reason": "Excellent machinability for micro/precision features"})
    slender_ratio = max_dim / mid_dim if mid_dim > 0 else 0
    if slender_ratio > 5 or "deep_feature" in issue_tags:
        processes.append("5-Axis CNC Milling")
        processes.append("Wire EDM (for deep slots)")
    else:
        processes.append("3-Axis CNC Milling")
    if mid_dim < 50 and min_dim < 20:
        processes.append("CNC Turning")
    if "thin_wall" in issue_tags:
        processes.append("Slow-feed Precision Milling")
    if geo["face_count"] > 50000:
        processes.append("5-Axis CNC Milling")
    return materials[:3], list(dict.fromkeys(processes))


def get_problem_faces(mesh, issues):
    problem_faces = set()
    normals = mesh.face_normals
    issue_tags = [i["tag"] for i in issues]
    if "slender" in issue_tags or "deep_feature" in issue_tags:
        centers = mesh.triangles_center
        main_axis = np.argmax(mesh.bounds[1] - mesh.bounds[0])
        axis_vals = centers[:, main_axis]
        threshold_low = np.percentile(axis_vals, 15)
        threshold_high = np.percentile(axis_vals, 85)
        for i, val in enumerate(axis_vals):
            if val < threshold_low or val > threshold_high:
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
            v = vertices[vi]
            positions.extend(v)
            if i in problem_set:
                colors.extend([0.9, 0.1, 0.1])
            else:
                colors.extend([0.45, 0.48, 0.52])
    return json.dumps({
        "positions": positions,
        "colors": colors,
        "face_count": len(faces),
        "vertex_count": len(vertices),
        "problem_face_count": len(problem_set)
    })


def render_3d_viewer(mesh_json_str, score):
    score_color = "#38a169" if score >= 85 else "#d69e2e" if score >= 60 else "#e53e3e"
    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d0d0d; overflow:hidden; }}
  canvas {{ display:block; }}
  #info {{ position:absolute; top:12px; left:12px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#555; line-height:1.8; }}
  #legend {{ position:absolute; bottom:12px; left:12px; font-family:'IBM Plex Mono',monospace; font-size:10px; color:#555; display:flex; gap:16px; align-items:center; }}
  .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }}
  #controls {{ position:absolute; top:12px; right:12px; display:flex; flex-direction:column; gap:6px; }}
  button {{ background:#1a1a1a; border:1px solid #2a2a2a; color:#777; font-family:'IBM Plex Mono',monospace; font-size:10px; padding:5px 10px; border-radius:4px; cursor:pointer; }}
  button:hover {{ background:#222; color:#aaa; }}
  #score-badge {{ position:absolute; top:12px; left:50%; transform:translateX(-50%); font-family:'IBM Plex Mono',monospace; font-size:11px; color:{score_color}; background:#0d0d0d; border:1px solid {score_color}44; border-radius:4px; padding:4px 12px; }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="info">drag to rotate · scroll to zoom</div>
<div id="score-badge">DFM Score: {score}/100</div>
<div id="legend">
  <span><span class="dot" style="background:#738085"></span>Normal</span>
  <span><span class="dot" style="background:#e53e3e"></span>Problem area</span>
</div>
<div id="controls">
  <button onclick="resetCamera()">Reset view</button>
  <button onclick="toggleWire()">Wireframe</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const data = {mesh_json_str};
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0d0d0d, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.01, 10000);
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.positions), 3));
geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(data.colors), 3));
geo.computeVertexNormals();
const mat = new THREE.MeshPhongMaterial({{vertexColors:true, shininess:40, specular:new THREE.Color(0x222222), side:THREE.DoubleSide}});
const wireMat = new THREE.MeshBasicMaterial({{color:0x334455, wireframe:true}});
const mesh = new THREE.Mesh(geo, mat);
scene.add(mesh);
geo.computeBoundingBox();
const center = new THREE.Vector3();
geo.boundingBox.getCenter(center);
mesh.position.sub(center);
const size = new THREE.Vector3();
geo.boundingBox.getSize(size);
const maxDim = Math.max(size.x, size.y, size.z);
camera.position.set(maxDim*1.2, maxDim*0.8, maxDim*1.5);
camera.lookAt(0,0,0);
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
dir1.position.set(1,2,3); scene.add(dir1);
const dir2 = new THREE.DirectionalLight(0x8899bb, 0.3);
dir2.position.set(-2,-1,-1); scene.add(dir2);
const grid = new THREE.GridHelper(maxDim*3, 20, 0x1a1a1a, 0x1a1a1a);
grid.position.y = -maxDim*0.6; scene.add(grid);
let isDown=false, lastX=0, lastY=0, rotX=0.3, rotY=0.5, zoom=1, wireframe=false;
canvas.addEventListener('mousedown', e=>{{isDown=true;lastX=e.clientX;lastY=e.clientY;}});
canvas.addEventListener('mouseup', ()=>isDown=false);
canvas.addEventListener('mousemove', e=>{{
  if(!isDown) return;
  rotY+=(e.clientX-lastX)*0.008; rotX+=(e.clientY-lastY)*0.008;
  lastX=e.clientX; lastY=e.clientY;
}});
canvas.addEventListener('wheel', e=>{{zoom*=e.deltaY>0?1.1:0.9;zoom=Math.max(0.2,Math.min(5,zoom));}});
function resetCamera(){{rotX=0.3;rotY=0.5;zoom=1;}}
function toggleWire(){{wireframe=!wireframe;mesh.material=wireframe?wireMat:mat;}}
function animate(){{
  requestAnimationFrame(animate);
  mesh.rotation.x=rotX; mesh.rotation.y=rotY;
  camera.position.set(maxDim*1.2*zoom,maxDim*0.8*zoom,maxDim*1.5*zoom);
  camera.lookAt(0,0,0); renderer.render(scene,camera);
}}
animate();
window.addEventListener('resize',()=>{{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
}});
</script></body></html>"""
    return html


def get_ai_advice(geo, issues, wall_data, undercut_data):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    issues_text = "\n".join([f"- [{i['severity'].upper()}] {i['title']}: {i['desc']}" for i in issues]) if issues else "- No major issues."
    wall_text = f"Min: {wall_data['min']:.2f}mm, Mean: {wall_data['mean']:.2f}mm" if wall_data else "Could not compute"
    undercut_text = f"Best axis: {undercut_data['best_axis']} ({undercut_data['best_pct']:.1f}% undercut). Multi-axis: {undercut_data['needs_multi_axis']}"
    prompt = f"""You are a senior DFM engineer at a precision CNC machining company.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Watertight: {geo['is_watertight']}
Wall thickness: {wall_text}
Undercut analysis: {undercut_text}

Issues:
{issues_text}

Write a concise assessment (max 200 words):
1. Overall manufacturability verdict
2. Most critical problem and real-world impact
3. Top 3 specific design changes with exact values
4. Recommended machining process and setup

Be direct. Use mm values. No filler sentences."""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def get_machining_guide(geo, issues, material):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    dims = sorted([geo["length"], geo["width"], geo["height"]])
    issue_tags = [i["tag"] for i in issues]
    prompt = f"""You are a master CNC machinist with 25 years experience. 

Part dimensions: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Material: {material}
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Issues detected: {', '.join(issue_tags) if issue_tags else 'none'}
Watertight mesh: {geo['is_watertight']}

Write a complete step-by-step machining guide from raw stock to finished part.
Include for each step:
- Step number and name
- Exact tool to use (drill size, end mill diameter, boring bar etc)
- Cutting parameters (RPM, feed rate, depth of cut)
- What to watch out for
- Quality check

Format as numbered steps. Be very specific with tool sizes and parameters.
Cover: raw stock selection, workholding setup, roughing, finishing, any holes/features, final inspection.
Max 600 words."""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def get_material_comparison(geo, materials_list, issues):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    issue_tags = [i["tag"] for i in issues]
    prompt = f"""You are a senior manufacturing engineer and materials specialist.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Issues: {', '.join(issue_tags) if issue_tags else 'none'}

Compare these materials for manufacturing this part:
{chr(10).join([f"- {m}" for m in materials_list])}

For each material provide:
1. Machinability score (1-10, 10=easiest)
2. Estimated difficulty for this specific part
3. Key challenges when machining
4. Surface finish achievable
5. Cost tier (Low/Medium/High/Very High)
6. Best use case for this material

End with a clear recommendation of which material to use and why.
Be specific. Use numbers where possible."""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def get_chat_response(messages):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=400,
        system="""You are a senior DFM (Design for Manufacturing) engineer with 20 years experience in CNC machining, turning, EDM, and precision manufacturing.
Answer questions concisely and practically. Use specific numbers and values.
Use line breaks between points. No markdown headers, no bullet symbols, just clean text.""",
        messages=messages
    )
    return response.content[0].text


def generate_pdf_report(geo, issues, score, cost_data, wall_data, undercut_data, ai_advice, filename="DFM_Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "DFM Pro - Manufacturing Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"File: {filename}", ln=True)
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Part Dimensions", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Length: {geo['length']:.1f} mm   Width: {geo['width']:.1f} mm   Height: {geo['height']:.1f} mm", ln=True)
    if geo["volume"]:
        pdf.cell(0, 6, f"Volume: {geo['volume']/1000:.2f} cm3   Surface Area: {geo['surface_area']:.1f} mm2", ln=True)
    pdf.cell(0, 6, f"Faces: {geo['face_count']:,}   Watertight: {'Yes' if geo['is_watertight'] else 'No'}", ln=True)
    pdf.ln(4)

    score_text = f"Manufacturability Score: {score}/100"
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, score_text, ln=True)
    verdict = "Ready to manufacture" if score >= 85 else "Needs minor revisions" if score >= 60 else "Significant redesign needed" if score >= 35 else "Not manufacturable as-is"
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Verdict: {verdict}", ln=True)
    pdf.cell(0, 6, f"Cost Tier: {['LOW','MEDIUM','HIGH','VERY HIGH'][0 if score>=85 else 1 if score>=60 else 2 if score>=35 else 3]}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Cost Estimate", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Material cost: Rs {cost_data['material_inr']:,}  |  Machining: Rs {cost_data['machining_inr']:,}  |  Setup: Rs {cost_data['setup_inr']:,}", ln=True)
    pdf.cell(0, 6, f"Total: Rs {cost_data['total_inr']:,}  (${cost_data['total_usd']:,})", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "DFM Issues", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if not issues:
        pdf.cell(0, 6, "No major issues detected.", ln=True)
    else:
        for issue in issues:
            pdf.cell(0, 6, f"[{issue['severity'].upper()}] {issue['title']}", ln=True)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, f"  {issue['desc']}", ln=True)
            pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    if wall_data:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Wall Thickness", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Min: {wall_data['min']:.2f}mm   Mean: {wall_data['mean']:.2f}mm   Max: {wall_data['max']:.2f}mm", ln=True)
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "AI Engineering Assessment", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for line in ai_advice.split("\n"):
        if line.strip():
            pdf.multi_cell(0, 5, line.strip())
    pdf.ln(2)

    output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    output.write(pdf_bytes)
    output.seek(0)
    return output


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


# ======== TAB 5 — SETTINGS (load first so thresholds available) ========
with tab5:
    st.markdown('<div class="section-title">Analysis Thresholds</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1rem">Adjust these to match your shop capabilities or client requirements.</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        thin_wall = st.slider("Thin wall warning threshold (mm)", 0.5, 5.0, 2.0, 0.1)
        critical_wall = st.slider("Critical wall threshold (mm)", 0.1, 2.0, 0.8, 0.1)
        depth_ratio = st.slider("Max depth/diameter ratio", 2, 15, 5, 1)
    with col_s2:
        max_size = st.slider("Max part size warning (mm)", 100, 2000, 500, 50)
        default_material = st.selectbox("Default material for cost estimate", [
            "Aluminium 6061", "Mild Steel (1018)", "Stainless Steel 304",
            "Titanium Grade 5", "Brass C360", "Free-machining Steel (12L14)", "Tool Steel D2"
        ])

    thresholds = {
        "thin_wall": thin_wall,
        "critical_wall": critical_wall,
        "depth_ratio": depth_ratio,
        "max_size": max_size
    }

    st.markdown('<div class="section-title">Currency</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.8rem">Exchange rate: 1 USD = 83.5 INR (fixed)</div>', unsafe_allow_html=True)


# ======== TAB 1 — STL ANALYZER ========
with tab1:
    uploaded_file = st.file_uploader("", type=["stl"], label_visibility="collapsed")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Loading geometry..."):
            mesh = trimesh.load(tmp_path)
            os.unlink(tmp_path)
            geo = analyze_geometry(mesh)

        issues = run_dfm_checks(geo, thresholds)
        score = calculate_score(issues)
        cost_tier, cost_color, cost_bg = get_cost_tier(score)
        cost_data = estimate_cost(geo, score, default_material)

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown('<div class="section-title">3D Model Viewer</div>', unsafe_allow_html=True)
            problem_faces = get_problem_faces(mesh, issues)
            mesh_json = mesh_to_threejs_json(mesh, problem_faces)
            viewer_html = render_3d_viewer(mesh_json, score)
            st.components.v1.html(viewer_html, height=480, scrolling=False)

            if problem_faces:
                st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#e53e3e;margin-top:0.5rem">⚠ {len(problem_faces)} problem faces highlighted in red</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#38a169;margin-top:0.5rem">✓ No problem areas detected</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Geometry</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Length</div><div class="metric-value">{geo["length"]:.1f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Width</div><div class="metric-value">{geo["width"]:.1f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Height</div><div class="metric-value">{geo["height"]:.1f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
            with c4:
                vol_display = f'{geo["volume"]/1000:.1f}' if geo["volume"] else "N/A"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Volume</div><div class="metric-value">{vol_display}</div><div class="metric-unit">cm3</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            pills = [
                f"Faces: {geo['face_count']:,}",
                f"Vertices: {geo['vertex_count']:,}",
                f"Surface: {geo['surface_area']:.0f} mm2",
                f"Watertight: {'Yes' if geo['is_watertight'] else 'No'}"
            ]
            if geo["fill_ratio"]:
                pills.append(f"Fill: {geo['fill_ratio']:.1f}%")
            pills_html = "".join([f'<span class="stat-pill">{p}</span>' for p in pills])
            st.markdown(f'<div class="stat-row">{pills_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Cost Estimate</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Cost (INR)</div><div class="metric-value" style="font-size:1.4rem">Rs {cost_data["total_inr"]:,}</div><div class="metric-unit">{default_material}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Cost (USD)</div><div class="metric-value" style="font-size:1.4rem">${cost_data["total_usd"]:,}</div><div class="metric-unit">approx</div></div>', unsafe_allow_html=True)
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

            st.markdown('<div class="section-title">Wall Thickness</div>', unsafe_allow_html=True)
            with st.spinner("Sampling..."):
                wall_data = estimate_wall_thickness(mesh)
            if wall_data:
                wc1, wc2, wc3 = st.columns(3)
                with wc1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Min</div><div class="metric-value">{wall_data["min"]:.2f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
                with wc2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Mean</div><div class="metric-value">{wall_data["mean"]:.2f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
                with wc3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Max</div><div class="metric-value">{wall_data["max"]:.2f}</div><div class="metric-unit">mm</div></div>', unsafe_allow_html=True)
                if wall_data["min"] < critical_wall:
                    st.markdown('<div class="issue-card issue-critical"><div class="issue-title">✕ Critical wall thickness</div><div class="issue-desc">Below critical threshold — extremely fragile.</div></div>', unsafe_allow_html=True)
                elif wall_data["min"] < thin_wall:
                    st.markdown('<div class="issue-card issue-warning"><div class="issue-title">⚠ Thin wall region</div><div class="issue-desc">Below threshold — consider increasing thickness.</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="issue-card issue-ok"><div class="issue-title">✓ Wall thickness acceptable</div><div class="issue-desc">All regions above threshold.</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Cannot compute</div><div class="issue-desc">Mesh too simple for ray sampling.</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Undercut Detection</div>', unsafe_allow_html=True)
            with st.spinner("Analyzing..."):
                undercut_data = detect_undercuts(mesh)
            uc1, uc2 = st.columns(2)
            with uc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Best axis</div><div class="metric-value" style="font-size:1rem">{undercut_data["best_axis"]}</div><div class="metric-unit">{undercut_data["best_pct"]:.1f}% undercut</div></div>', unsafe_allow_html=True)
            with uc2:
                multi = "5-axis needed" if undercut_data["needs_multi_axis"] else "3-axis OK"
                uc_color = "#e53e3e" if undercut_data["needs_multi_axis"] else "#38a169"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Setup</div><div class="metric-value" style="font-size:1rem;color:{uc_color}">{multi}</div></div>', unsafe_allow_html=True)

            for axis, data in undercut_data["axes"].items():
                bar_color = "#38a169" if data["undercut_pct"] < 15 else "#d69e2e" if data["undercut_pct"] < 35 else "#e53e3e"
                st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.35rem"><span style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#555;width:90px">{axis}</span><div style="flex:1;background:#1a1a1a;border-radius:3px;height:5px"><div style="width:{min(data["undercut_pct"],100):.1f}%;background:{bar_color};height:5px;border-radius:3px"></div></div><span style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#555;width:38px;text-align:right">{data["undercut_pct"]:.1f}%</span></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">DFM Issues</div>', unsafe_allow_html=True)
            if not issues:
                st.markdown('<div class="issue-card issue-ok"><div class="issue-title">✓ No issues detected</div><div class="issue-desc">Part meets manufacturability criteria.</div></div>', unsafe_allow_html=True)
            else:
                severity_order = {"critical": 0, "warning": 1, "info": 2}
                for issue in sorted(issues, key=lambda x: severity_order.get(x["severity"], 3)):
                    css_class = {"critical": "issue-critical", "warning": "issue-warning", "info": "issue-info"}.get(issue["severity"], "issue-info")
                    icon = {"critical": "✕", "warning": "⚠", "info": "ℹ"}.get(issue["severity"], "•")
                    st.markdown(f'<div class="issue-card {css_class}"><div class="issue-title">{icon} {issue["title"]}</div><div class="issue-desc">{issue["desc"]}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Material & Process</div>', unsafe_allow_html=True)
            materials, processes = recommend_materials_and_process(geo, issues)
            for mat in materials:
                st.markdown(f'<div class="mat-card"><div class="mat-name">{mat["name"]}</div><div class="mat-reason">{mat["reason"]}</div></div>', unsafe_allow_html=True)
            badges = "".join([f'<span class="process-badge">{p}</span>' for p in processes])
            st.markdown(f'<div style="margin-top:0.3rem">{badges}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">AI Engineering Assessment</div>', unsafe_allow_html=True)
            with st.spinner("Generating assessment..."):
                advice = get_ai_advice(geo, issues, wall_data, undercut_data)
            st.markdown(f'<div class="ai-box"><div class="ai-label">▸ CLAUDE AI — MANUFACTURING ENGINEER</div>{advice.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Export Report</div>', unsafe_allow_html=True)
            if st.button("Generate PDF Report", key="pdf_btn"):
                with st.spinner("Generating PDF..."):
                    pdf_buffer = generate_pdf_report(
                        geo, issues, score, cost_data,
                        wall_data, undercut_data, advice,
                        uploaded_file.name
                    )
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"DFM_Report_{uploaded_file.name.replace('.stl','')}.pdf",
                    mime="application/pdf"
                )

        st.session_state["geo"] = geo
        st.session_state["issues"] = issues
        st.session_state["mesh_loaded"] = True
        st.session_state["filename"] = uploaded_file.name

    else:
        st.session_state["mesh_loaded"] = False
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3">⚙</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:#333">Upload an STL file to begin analysis</div>
            <div style="font-size:0.8rem;color:#2a2a2a;margin-top:0.5rem">Supports binary and ASCII STL · drag and drop</div>
        </div>
        """, unsafe_allow_html=True)


# ======== TAB 2 — MACHINING GUIDE ========
with tab2:
    st.markdown('<div class="section-title">Complete Machining Guide</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1.5rem">Step-by-step guide from raw stock to finished part — tools, parameters, and what to watch out for.</div>', unsafe_allow_html=True)

    if not st.session_state.get("mesh_loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first</div><div class="issue-desc">Go to STL Analyzer tab and upload your file, then come back here.</div></div>', unsafe_allow_html=True)
    else:
        geo = st.session_state["geo"]
        issues = st.session_state["issues"]

        guide_material = st.selectbox("Material to machine in", [
            "Aluminium 6061", "Mild Steel (1018)", "Stainless Steel 304",
            "Titanium Grade 5", "Brass C360", "Free-machining Steel (12L14)", "Tool Steel D2"
        ], key="guide_material")

        if st.button("Generate Machining Guide", key="guide_btn"):
            with st.spinner("Generating complete machining guide..."):
                guide = get_machining_guide(geo, issues, guide_material)
            st.session_state["machining_guide"] = guide

        if "machining_guide" in st.session_state:
            guide_text = st.session_state["machining_guide"]
            lines = guide_text.strip().split("\n")
            current_step = []
            step_num = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                is_new_step = (line[0].isdigit() and ("." in line[:3] or ")" in line[:3]))
                if is_new_step and current_step:
                    title = current_step[0]
                    detail = " ".join(current_step[1:])
                    step_num += 1
                    st.markdown(f'<div class="step-card"><div class="step-num">STEP {step_num}</div><div class="step-title">{title}</div><div class="step-detail">{detail}</div></div>', unsafe_allow_html=True)
                    current_step = [line]
                elif is_new_step:
                    current_step = [line]
                else:
                    current_step.append(line)

            if current_step:
                title = current_step[0]
                detail = " ".join(current_step[1:])
                step_num += 1
                st.markdown(f'<div class="step-card"><div class="step-num">STEP {step_num}</div><div class="step-title">{title}</div><div class="step-detail">{detail}</div></div>', unsafe_allow_html=True)


# ======== TAB 3 — MATERIAL COMPARATOR ========
with tab3:
    st.markdown('<div class="section-title">Material Comparator</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1.5rem">Enter materials you are considering. AI will compare machinability, cost, and difficulty for your specific part.</div>', unsafe_allow_html=True)

    if not st.session_state.get("mesh_loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first</div><div class="issue-desc">Go to STL Analyzer tab and upload your file, then come back here.</div></div>', unsafe_allow_html=True)
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Select from common materials**")
            common_materials = [
                "Aluminium 6061", "Aluminium 7075", "Mild Steel (1018)",
                "Stainless Steel 304", "Stainless Steel 316", "Titanium Grade 5",
                "Brass C360", "Copper", "Free-machining Steel (12L14)",
                "Tool Steel D2", "Inconel 718", "Delrin (POM)"
            ]
            selected_common = st.multiselect("", common_materials, default=["Aluminium 6061", "Mild Steel (1018)"])

        with col_m2:
            st.markdown("**Or add custom materials**")
            custom_mat1 = st.text_input("Custom material 1", placeholder="e.g. Bronze C932")
            custom_mat2 = st.text_input("Custom material 2", placeholder="e.g. PEEK plastic")
            custom_mat3 = st.text_input("Custom material 3", placeholder="e.g. Aluminium 2024")

        all_materials = selected_common.copy()
        for m in [custom_mat1, custom_mat2, custom_mat3]:
            if m.strip():
                all_materials.append(m.strip())

        if len(all_materials) < 2:
            st.markdown('<div class="issue-card issue-warning"><div class="issue-title">⚠ Select at least 2 materials</div><div class="issue-desc">Add more materials to compare.</div></div>', unsafe_allow_html=True)
        else:
            if st.button("Compare Materials", key="compare_btn"):
                with st.spinner(f"Comparing {len(all_materials)} materials..."):
                    comparison = get_material_comparison(
                        st.session_state["geo"],
                        all_materials,
                        st.session_state["issues"]
                    )
                st.session_state["material_comparison"] = comparison
                st.session_state["compared_materials"] = all_materials

        if "material_comparison" in st.session_state:
            st.markdown('<div class="section-title">Comparison Results</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box"><div class="ai-label">▸ CLAUDE AI — MATERIALS ENGINEER</div>{st.session_state["material_comparison"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Cost Comparison</div>', unsafe_allow_html=True)
            geo = st.session_state["geo"]
            issues = st.session_state["issues"]
            score = calculate_score(issues)
            cost_cols = st.columns(min(len(st.session_state["compared_materials"]), 4))
            for i, mat in enumerate(st.session_state["compared_materials"][:4]):
                cost = estimate_cost(geo, score, mat)
                with cost_cols[i]:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">{mat[:20]}</div><div class="metric-value" style="font-size:1.1rem">Rs {cost["total_inr"]:,}</div><div class="metric-unit">${cost["total_usd"]}</div></div>', unsafe_allow_html=True)


# ======== TAB 4 — DFM CHAT ========
with tab4:
    st.markdown('<div class="section-title">DFM Engineering Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1.5rem">Ask any manufacturing question — tolerances, materials, processes, costs, tooling.</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai"><div class="chat-ai-bubble">{msg["content"].replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)

    if len(st.session_state.chat_history) == 0:
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
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    reply = get_chat_response([{"role": "user", "content": suggestion}])
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

    user_input = st.chat_input("Ask a manufacturing question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
        reply = get_chat_response(messages)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if len(st.session_state.chat_history) > 0:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
