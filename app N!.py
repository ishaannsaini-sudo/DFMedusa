import streamlit as st
import trimesh
import tempfile
import os
import anthropic
import numpy as np
import json
import math

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
    .geo-box { background: #0d1117; border: 1px solid #1e1e1e; border-left: 3px solid #d69e2e; border-radius: 8px; padding: 1.2rem; font-size: 0.875rem; color: #c0c0c0; line-height: 1.7; }
    .fea-box { background: #0d1117; border: 1px solid #1e1e1e; border-left: 3px solid #38a169; border-radius: 8px; padding: 1.2rem; font-size: 0.875rem; color: #c0c0c0; line-height: 1.7; }
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
    .part-type-badge { display:inline-block; padding:0.3rem 0.8rem; border-radius:20px; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; font-weight:600; margin-bottom:0.5rem; }
    .fea-result { background:#111; border:1px solid #1e1e1e; border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:0.5rem; }
    .fea-label { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#555; margin-bottom:0.3rem; }
    .fea-value { font-family:'IBM Plex Mono',monospace; font-size:1.2rem; font-weight:600; }
    .stress-ok { color: #68d391; }
    .stress-warn { color: #f6e05e; }
    .stress-crit { color: #fc8181; }
    .step-card { background:#111; border:1px solid #1e1e1e; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.6rem; border-left:3px solid #3182ce; }
    .step-num { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#3182ce; letter-spacing:0.1em; margin-bottom:0.3rem; }
    .step-title { font-size:0.9rem; font-weight:600; color:#e0e0e0; margin-bottom:0.3rem; }
    .step-detail { font-size:0.78rem; color:#888; line-height:1.6; }
    .part-type-selector { background:#111; border:1px solid #2a2a2a; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; }
    .type-option { background:#0d0d0d; border:1px solid #1e1e1e; border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem; cursor:pointer; transition:border-color 0.15s; }
    .type-option:hover { border-color:#3182ce; }
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


def generate_part_summary(geo, issues, part_type):
    issue_text = ", ".join([i["msg"] for i in issues]) if issues else "none"
    prompt = f"""You are a senior DFM engineer analyzing a {part_type}.

Dimensions: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Aspect ratio: {geo['aspect_ratio']:.1f}:1
Watertight: {geo['watertight']}
Issues found: {issue_text}

Write a 3-sentence engineering summary specific to {part_type} manufacturing:
1. What kind of part this is and its geometry characteristics
2. Key manufacturability considerations for {part_type}
3. Primary recommendation

Be specific and direct. Reference actual dimensions."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=200)


def get_geometry_optimization(geo, part_type, issues):
    issue_tags = [i.get("tag", "") for i in issues]
    prompt = f"""You are a structural engineering and DFM expert specializing in {part_type} design optimization.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Type: {part_type}
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Aspect ratio: {geo['aspect_ratio']:.1f}:1
Issues: {', '.join(issue_tags) if issue_tags else 'none'}

Suggest 4-5 specific geometry changes to improve strength and reduce cost. For each suggestion:
1. What to change (specific feature with exact dimensions)
2. Why it helps (structural principle — like corrugation increases second moment of area)
3. Expected benefit (% cost reduction or % strength increase)
4. Real-world example (like corrugated roof sheets, I-beams, bicycle frames)

Think like a structural engineer. Suggest ribs, gussets, corrugations, flanges, relief cuts, chamfers, fillets, hollow sections, lattice patterns based on what this part actually needs.
Be specific with mm values. No generic advice."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=700)


def get_stress_assessment(geo, part_type, stress_data, material):
    prompt = f"""You are a structural engineer. Assess stress risks for this {part_type}.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Material: {material}
Aspect ratio: {geo['aspect_ratio']:.1f}:1

Geometric stress indicators found:
- Sharp corner regions: {stress_data['sharp_corners']} detected
- Thin section changes: {stress_data['thin_sections']} detected
- High curvature areas: {stress_data['high_curvature']} detected
- Stress concentration factor estimate: {stress_data['kt_estimate']:.1f}

For each stress risk, explain:
1. Where stress concentrates and why
2. Failure mode (fatigue, yielding, buckling)
3. Exact fix (fillet radius, relief cut depth, rib placement with mm values)

Be specific. Reference actual geometry. No generic answers."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=600)


def get_ai_advice(geo, issues, part_type):
    issue_text = "\n".join([f"- {i['msg']}" for i in issues]) if issues else "- No major issues"
    prompt = f"""You are a senior DFM engineer specializing in {part_type}.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Type: {part_type}
Volume: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
Aspect ratio: {geo['aspect_ratio']:.1f}:1

Issues:
{issue_text}

Write concise {part_type} specific assessment (max 180 words):
1. Manufacturability verdict for {part_type}
2. Most critical issue specific to {part_type} manufacturing
3. Top 3 fixes with exact values
4. Recommended process for {part_type}

Direct, specific, use mm values."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=400)


def get_context_chat_response(chat_history, part_context):
    system = f"""You are a senior DFM engineer with 20 years experience.

The engineer has uploaded a part with these exact specifications:
{part_context}

RULES:
- Every answer must be specific to THIS part
- Reference actual dimensions and features
- Never give generic textbook answers
- Use mm values from the actual part
- Keep answers concise and practical
- Answer based on the part type specified"""

    messages = []
    for m in chat_history:
        if m["role"] in ["user", "assistant"]:
            messages.append({"role": m["role"], "content": str(m["content"])})
    if not messages:
        return "Please ask a question about your part."
    return safe_ai(messages, system=system, max_tokens=500)


def get_machining_guide(geo, material, part_type):
    prompt = f"""You are a master machinist with 25 years experience in {part_type} manufacturing.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Material: {material}
Type: {part_type}

Write complete step-by-step {part_type} manufacturing guide.
For each step: exact tool, parameters, what to watch for.
Cover: raw stock, setup, primary operations, finishing, inspection.
Max 500 words. Very specific to {part_type} process."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=700)


def get_material_comparison(geo, materials_list, part_type, issues):
    issue_tags = [i.get("tag", "") for i in issues]
    prompt = f"""You are a senior materials engineer specializing in {part_type}.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Type: {part_type}
Issues: {', '.join(issue_tags) if issue_tags else 'none'}

Compare these materials specifically for {part_type} manufacturing:
{chr(10).join([f"- {m}" for m in materials_list])}

For each material:
1. Machinability/formability score for {part_type} (1-10)
2. Suitability for this specific geometry
3. Key {part_type} manufacturing challenges
4. Cost tier

End with clear recommendation for {part_type}. Be specific."""
    return safe_ai([{"role": "user", "content": prompt}], max_tokens=700)


# ==================== GEOMETRY + PHYSICS FUNCTIONS ====================

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
        "bbox_volume": bbox_volume,
        "dims_sorted": dims
    }


def detect_stress_indicators(mesh, geo):
    normals = mesh.face_normals
    total_faces = len(normals)
    stress_data = {
        "sharp_corners": 0,
        "thin_sections": 0,
        "high_curvature": 0,
        "kt_estimate": 1.0,
        "problem_faces": set(),
        "stress_faces": set()
    }

    try:
        # Detect sharp corners — faces with high dihedral angle changes
        edges = mesh.face_adjacency
        edge_angles = mesh.face_adjacency_angles

        sharp_threshold = np.radians(120)
        sharp_edge_mask = edge_angles > sharp_threshold
        sharp_edges = edges[sharp_edge_mask]

        sharp_face_set = set()
        for edge_pair in sharp_edges:
            sharp_face_set.add(int(edge_pair[0]))
            sharp_face_set.add(int(edge_pair[1]))

        stress_data["sharp_corners"] = len(sharp_edges)
        stress_data["stress_faces"] = sharp_face_set

        # Detect thin sections using face area variation
        face_areas = mesh.area_faces
        mean_area = np.mean(face_areas)
        std_area = np.std(face_areas)
        thin_mask = face_areas < (mean_area - 1.5 * std_area)
        stress_data["thin_sections"] = int(np.sum(thin_mask))

        thin_indices = np.where(thin_mask)[0]
        for idx in thin_indices[:200]:
            stress_data["stress_faces"].add(int(idx))

        # Detect high curvature areas
        if hasattr(mesh, 'vertex_normals'):
            vertex_normals = mesh.vertex_normals
            faces = mesh.faces
            curvature_count = 0
            for face in faces[:min(500, len(faces))]:
                face_vn = vertex_normals[face]
                dots = np.dot(face_vn[0], face_vn[1]), np.dot(face_vn[1], face_vn[2])
                avg_dot = (dots[0] + dots[1]) / 2
                if avg_dot < 0.85:
                    curvature_count += 1
            stress_data["high_curvature"] = curvature_count

        # Estimate stress concentration factor Kt
        kt = 1.0
        if stress_data["sharp_corners"] > 10:
            kt += 0.5
        if stress_data["sharp_corners"] > 50:
            kt += 0.8
        if geo["aspect_ratio"] > 5:
            kt += 0.3
        if geo["fill_ratio"] and geo["fill_ratio"] < 30:
            kt += 0.2
        stress_data["kt_estimate"] = round(kt, 1)

    except Exception:
        pass

    return stress_data


def simplified_fea(geo, part_type, material, load_n=1000):
    dims = geo["dims_sorted"]
    min_dim, mid_dim, max_dim = dims[0], dims[1], dims[2]

    material_props = {
        "Aluminium 6061": {"E": 69e3, "yield": 276, "density": 2.7},
        "Mild Steel (1018)": {"E": 200e3, "yield": 370, "density": 7.85},
        "Stainless Steel 304": {"E": 193e3, "yield": 215, "density": 8.0},
        "Titanium Grade 5": {"E": 114e3, "yield": 880, "density": 4.43},
        "Brass C360": {"E": 97e3, "yield": 124, "density": 8.5},
        "Free-machining Steel (12L14)": {"E": 200e3, "yield": 414, "density": 7.85},
        "Tool Steel D2": {"E": 210e3, "yield": 1520, "density": 7.7},
        "Polypropylene": {"E": 1.4e3, "yield": 35, "density": 0.91},
        "ABS Plastic": {"E": 2.3e3, "yield": 40, "density": 1.05},
        "Nylon PA66": {"E": 3.0e3, "yield": 85, "density": 1.14},
    }

    props = material_props.get(material, {"E": 200e3, "yield": 370, "density": 7.85})
    E = props["E"]
    yield_strength = props["yield"]
    density = props["density"]

    results = {}

    L = max_dim / 1000
    b = mid_dim / 1000
    h = min_dim / 1000

    # Mass
    if geo["volume"]:
        volume_m3 = geo["volume"] * 1e-9
    else:
        volume_m3 = L * b * h * 0.6
    mass_kg = volume_m3 * density * 1000
    results["mass_kg"] = round(mass_kg, 3)

    # Beam bending (cantilever — worst case)
    I = (b * h**3) / 12
    if I > 0:
        max_bending_stress = (load_n * L * (h / 2)) / I / 1e6
        deflection_mm = (load_n * L**3) / (3 * E * 1e6 * I) * 1000
        results["bending_stress_mpa"] = round(max_bending_stress, 1)
        results["deflection_mm"] = round(deflection_mm, 3)
        results["safety_factor_bending"] = round(yield_strength / max(max_bending_stress, 0.01), 2)
    else:
        results["bending_stress_mpa"] = 0
        results["deflection_mm"] = 0
        results["safety_factor_bending"] = 999

    # Direct axial stress
    area_m2 = b * h
    axial_stress = load_n / area_m2 / 1e6
    results["axial_stress_mpa"] = round(axial_stress, 2)
    results["safety_factor_axial"] = round(yield_strength / max(axial_stress, 0.01), 1)

    # Buckling (Euler) for slender parts
    if geo["aspect_ratio"] > 3:
        A = b * h
        Le = L
        r_gyration = math.sqrt(I / A) if A > 0 else 0.001
        slenderness = Le / r_gyration if r_gyration > 0 else 999
        if slenderness > 0:
            euler_load = (math.pi**2 * E * 1e6 * I) / (Le**2)
            buckling_sf = euler_load / max(load_n, 1)
            results["buckling_load_n"] = round(euler_load, 0)
            results["safety_factor_buckling"] = round(buckling_sf, 1)
            results["slenderness_ratio"] = round(slenderness, 1)
        else:
            results["buckling_load_n"] = 999999
            results["safety_factor_buckling"] = 999
            results["slenderness_ratio"] = 0
    else:
        results["buckling_load_n"] = None
        results["safety_factor_buckling"] = None
        results["slenderness_ratio"] = None

    # Natural frequency (first mode approximation)
    if geo["volume"] and mass_kg > 0:
        k_approx = (3 * E * 1e6 * I) / (L**3) if L > 0 else 1
        omega = math.sqrt(k_approx / max(mass_kg, 0.001))
        freq_hz = omega / (2 * math.pi)
        results["natural_freq_hz"] = round(freq_hz, 1)
    else:
        results["natural_freq_hz"] = None

    results["load_applied_n"] = load_n
    results["material"] = material
    results["E_gpa"] = E / 1000
    results["yield_mpa"] = yield_strength

    return results


def run_dfm_checks(geo, thresholds, part_type):
    issues = []
    dims = geo["dims_sorted"]
    min_dim, mid_dim, max_dim = dims[0], dims[1], dims[2]

    # Part-type specific checks
    if part_type == "Sheet Metal":
        if min_dim < 0.3:
            issues.append({"severity": "critical", "msg": f"Sheet too thin {min_dim:.2f}mm — will tear during forming", "penalty": 35, "tag": "thin_wall"})
        elif min_dim < 0.8:
            issues.append({"severity": "warning", "msg": f"Thin sheet {min_dim:.2f}mm — check bend radius", "penalty": 20, "tag": "thin_wall"})

        ratio = max_dim / max(mid_dim, 0.01)
        if ratio > 20:
            issues.append({"severity": "warning", "msg": f"Long narrow sheet {ratio:.1f}:1 — risk of warping during forming", "penalty": 15, "tag": "slender"})

        if max_dim > 2000:
            issues.append({"severity": "warning", "msg": f"Large sheet {max_dim:.0f}mm — may exceed press brake capacity", "penalty": 15, "tag": "oversized"})

        if min_dim > 6:
            issues.append({"severity": "info", "msg": f"Thick sheet {min_dim:.1f}mm — laser cutting may not be cost effective, consider waterjet", "penalty": 5, "tag": "process"})

    elif part_type == "Injection Molded Plastic":
        if min_dim < 0.8:
            issues.append({"severity": "critical", "msg": f"Wall too thin {min_dim:.2f}mm — plastic will not fill mold", "penalty": 35, "tag": "thin_wall"})
        elif min_dim < 1.5:
            issues.append({"severity": "warning", "msg": f"Thin wall {min_dim:.2f}mm — risk of short shots and sink marks", "penalty": 20, "tag": "thin_wall"})

        ratio = max_dim / max(min_dim, 0.01)
        if ratio > 8:
            issues.append({"severity": "warning", "msg": f"High aspect ratio {ratio:.1f}x — flow length may be too long for consistent fill", "penalty": 20, "tag": "deep_feature"})

        if geo["fill_ratio"] and geo["fill_ratio"] > 80:
            issues.append({"severity": "info", "msg": "High material volume — consider coring out to reduce sink marks and cycle time", "penalty": 10, "tag": "hollow"})

        if max_dim > 500:
            issues.append({"severity": "warning", "msg": f"Large part {max_dim:.0f}mm — requires large tonnage press, high tooling cost", "penalty": 15, "tag": "oversized"})

    elif part_type == "Sand Casting":
        if min_dim < 3:
            issues.append({"severity": "critical", "msg": f"Section too thin {min_dim:.2f}mm — metal will solidify before filling", "penalty": 35, "tag": "thin_wall"})
        elif min_dim < 6:
            issues.append({"severity": "warning", "msg": f"Thin cast section {min_dim:.2f}mm — risk of cold shuts", "penalty": 20, "tag": "thin_wall"})

        if max_dim > 1000:
            issues.append({"severity": "warning", "msg": f"Large casting {max_dim:.0f}mm — significant pattern and flask costs", "penalty": 15, "tag": "oversized"})

        if geo["aspect_ratio"] > 6:
            issues.append({"severity": "warning", "msg": f"High aspect ratio {geo['aspect_ratio']:.1f}:1 — core may shift during pouring", "penalty": 20, "tag": "slender"})

    elif part_type == "Welded Assembly":
        if min_dim < 1.5:
            issues.append({"severity": "critical", "msg": f"Section too thin {min_dim:.2f}mm — will burn through during welding", "penalty": 35, "tag": "thin_wall"})
        elif min_dim < 3:
            issues.append({"severity": "warning", "msg": f"Thin section {min_dim:.2f}mm — requires skilled welder and heat control", "penalty": 20, "tag": "thin_wall"})

        if geo["aspect_ratio"] > 8:
            issues.append({"severity": "warning", "msg": f"Slender assembly {geo['aspect_ratio']:.1f}:1 — welding distortion risk", "penalty": 20, "tag": "slender"})

    else:
        # CNC Machined Solid — original rules
        if min_dim < thresholds["critical_wall"]:
            issues.append({"severity": "critical", "msg": f"Extremely thin feature {min_dim:.2f}mm — impossible to machine", "penalty": 35, "tag": "thin_wall"})
        elif min_dim < thresholds["thin_wall"]:
            issues.append({"severity": "warning", "msg": f"Thin wall {min_dim:.2f}mm — risk of breakage during machining", "penalty": 20, "tag": "thin_wall"})

        ratio = max_dim / max(min_dim, 0.01)
        if ratio > thresholds["depth_ratio"] * 2:
            issues.append({"severity": "critical", "msg": f"Extreme depth ratio {ratio:.1f}x — standard tooling cannot reach", "penalty": 35, "tag": "deep_feature"})
        elif ratio > thresholds["depth_ratio"]:
            issues.append({"severity": "warning", "msg": f"Deep feature ratio {ratio:.1f}x (limit: {thresholds['depth_ratio']}x)", "penalty": 20, "tag": "deep_feature"})

        if max_dim > thresholds["max_size"]:
            issues.append({"severity": "warning", "msg": f"Large part {max_dim:.1f}mm — high machining cost", "penalty": 15, "tag": "oversized"})

        slender = max_dim / max(mid_dim, 0.01)
        if slender > 10:
            issues.append({"severity": "critical", "msg": f"Extremely slender {slender:.1f}:1 — severe deflection risk", "penalty": 30, "tag": "slender"})
        elif slender > 5:
            issues.append({"severity": "warning", "msg": f"Slender geometry {slender:.1f}:1 — may deflect during machining", "penalty": 15, "tag": "slender"})

    # Common checks for all types
    if not geo["watertight"]:
        issues.append({"severity": "warning", "msg": "Non-watertight mesh — check for missing faces", "penalty": 10, "tag": "mesh"})
    if geo["faces"] > 100000:
        issues.append({"severity": "info", "msg": f"High complexity {geo['faces']:,} faces", "penalty": 5, "tag": "complexity"})

    return issues


def calculate_score(issues):
    return max(0, min(100, 100 - sum(i["penalty"] for i in issues)))


def get_cost_tier(score):
    if score >= 85: return "LOW", "#38a169", "#0a1a0f"
    elif score >= 60: return "MEDIUM", "#d69e2e", "#1a1400"
    elif score >= 35: return "HIGH", "#e53e3e", "#1a0a0a"
    else: return "VERY HIGH", "#9b2c2c", "#150505"


def estimate_cost(geo, score, material, part_type):
    volume_cm3 = (geo["volume"] / 1000) if geo["volume"] else (geo["bbox_volume"] / 1000 * 0.6)

    # Material rates by type
    if part_type == "Sheet Metal":
        rates = {"Aluminium 6061": 280, "Mild Steel (1018)": 150, "Stainless Steel 304": 420, "Titanium Grade 5": 3000}
        base_machining = 400
    elif part_type == "Injection Molded Plastic":
        rates = {"Polypropylene": 80, "ABS Plastic": 120, "Nylon PA66": 200}
        base_machining = 200
    elif part_type == "Sand Casting":
        rates = {"Aluminium 6061": 200, "Mild Steel (1018)": 120, "Brass C360": 500}
        base_machining = 600
    else:
        rates = {"Aluminium 6061": 350, "Mild Steel (1018)": 180, "Stainless Steel 304": 520, "Titanium Grade 5": 3500, "Brass C360": 650, "Free-machining Steel (12L14)": 220, "Tool Steel D2": 800}
        base_machining = 800

    rate = rates.get(material, 300)
    material_cost = volume_cm3 * rate * 0.00785
    machining_cost = volume_cm3 * base_machining * (1.0 + (100 - score) / 100 * 2.5)
    setup_cost = 2500 if score > 70 else 5000 if score > 40 else 9000

    if part_type == "Injection Molded Plastic":
        setup_cost += 50000  # Mold cost amortized

    total_inr = material_cost + machining_cost + setup_cost
    return {
        "material_inr": round(material_cost),
        "machining_inr": round(machining_cost),
        "setup_inr": round(setup_cost),
        "total_inr": round(total_inr),
        "total_usd": round(total_inr / 83.5, 2)
    }


def get_problem_faces(mesh, issues, stress_data):
    problem_faces = set(stress_data.get("stress_faces", set()))
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


def mesh_to_threejs_json(mesh, problem_face_indices, stress_faces):
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    problem_set = set(problem_face_indices)
    stress_set = set(list(stress_faces)[:500]) if stress_faces else set()
    positions, colors = [], []
    for i, face in enumerate(faces):
        for vi in face:
            positions.extend(vertices[vi])
            if i in problem_set and i in stress_set:
                colors.extend([1.0, 0.2, 0.2])   # bright red — critical
            elif i in stress_set:
                colors.extend([0.9, 0.5, 0.1])   # orange — stress indicator
            elif i in problem_set:
                colors.extend([0.85, 0.1, 0.1])  # red — DFM issue
            else:
                colors.extend([0.45, 0.48, 0.52]) # grey — normal
    return json.dumps({"positions": positions, "colors": colors,
                       "problem_count": len(problem_set), "stress_count": len(stress_set)})


def render_3d_viewer(mesh_json_str, score):
    score_color = "#38a169" if score >= 85 else "#d69e2e" if score >= 60 else "#e53e3e"
    return f"""<!DOCTYPE html><html><head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d0d0d; overflow:hidden; }}
  canvas {{ display:block; }}
  #info {{ position:absolute; top:12px; left:12px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#555; }}
  #legend {{ position:absolute; bottom:12px; left:12px; font-family:'IBM Plex Mono',monospace; font-size:10px; color:#555; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
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
  <span><span class="dot" style="background:#e88020"></span>Stress zone</span>
  <span><span class="dot" style="background:#e53e3e"></span>DFM issue</span>
  <span><span class="dot" style="background:#ff3030"></span>Critical</span>
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


def build_part_context(geo, issues, score, cost_data, part_type, fea_results, stress_data):
    issue_list = "\n".join([f"  - [{i['severity'].upper()}] {i['msg']}" for i in issues]) if issues else "  - None"
    fea_text = ""
    if fea_results:
        fea_text = f"""
STRUCTURAL ANALYSIS (simplified):
  - Bending stress: {fea_results.get('bending_stress_mpa', 'N/A')} MPa (yield: {fea_results.get('yield_mpa', 'N/A')} MPa)
  - Safety factor (bending): {fea_results.get('safety_factor_bending', 'N/A')}
  - Deflection at 1000N: {fea_results.get('deflection_mm', 'N/A')} mm
  - Mass: {fea_results.get('mass_kg', 'N/A')} kg"""

    return f"""PART TYPE: {part_type}
DIMENSIONS: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
VOLUME: {f"{geo['volume']:.1f} mm3" if geo['volume'] else 'N/A'}
ASPECT RATIO: {geo['aspect_ratio']:.1f}:1
WATERTIGHT: {geo['watertight']}
SCORE: {score}/100

DFM ISSUES:
{issue_list}

STRESS INDICATORS:
  - Sharp corners detected: {stress_data.get('sharp_corners', 0)}
  - Thin sections: {stress_data.get('thin_sections', 0)}
  - Stress concentration factor Kt: {stress_data.get('kt_estimate', 1.0)}
{fea_text}

COST ({cost_data.get('material', 'default material')}): Rs {cost_data['total_inr']:,} (${cost_data['total_usd']})"""


# ==================== UI ====================

st.markdown('<div class="hero-title">⚙ DFM Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered Design for Manufacturing — Part type aware · Stress analysis · Geometry optimization · Simplified FEA</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁  Analyzer",
    "📐  FEA & Stress",
    "💡  Geometry Optimizer",
    "🔧  Machining Guide",
    "💬  Part Chat",
    "⚙  Settings"
])


# ==================== SETTINGS ====================
with tab6:
    st.markdown('<div class="section-title">Analysis Thresholds (CNC Machined)</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        thin_wall = st.slider("Thin wall warning (mm)", 0.5, 5.0, 2.0, 0.1)
        critical_wall = st.slider("Critical wall (mm)", 0.1, 2.0, 0.8, 0.1)
        depth_ratio = st.slider("Max depth/diameter ratio", 2, 15, 5)
    with col_s2:
        max_size = st.slider("Max part size warning (mm)", 100, 2000, 500, 50)
        default_material = st.selectbox("Default material", [
            "Aluminium 6061", "Mild Steel (1018)", "Stainless Steel 304",
            "Titanium Grade 5", "Brass C360", "Free-machining Steel (12L14)",
            "Tool Steel D2", "Polypropylene", "ABS Plastic", "Nylon PA66"
        ])

thresholds = {"thin_wall": thin_wall, "critical_wall": critical_wall, "depth_ratio": depth_ratio, "max_size": max_size}


# ==================== ANALYZER ====================
with tab1:

    # STEP 1 — Part type selector
    st.markdown('<div class="section-title">Step 1 — What type of part is this?</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1rem">Select part type for accurate DFM analysis. Rules change completely based on manufacturing process.</div>', unsafe_allow_html=True)

    part_type_options = {
        "CNC Machined Solid": "🔩  CNC Machined Solid — Aluminium, steel, brass cut from billet",
        "Sheet Metal": "📄  Sheet Metal — Laser cut, bent, stamped metal sheet",
        "Injection Molded Plastic": "🧴  Injection Molded Plastic — Plastic parts from a mold",
        "Sand Casting": "🏭  Sand Casting — Poured metal into sand mold",
        "Welded Assembly": "⚡  Welded Assembly — Multiple parts joined by welding",
        "3D Printed": "🖨  3D Printed — FDM, SLS, SLA additive manufacturing"
    }

    part_type_colors = {
        "CNC Machined Solid": "#3182ce",
        "Sheet Metal": "#d69e2e",
        "Injection Molded Plastic": "#38a169",
        "Sand Casting": "#e53e3e",
        "Welded Assembly": "#805ad5",
        "3D Printed": "#dd6b20"
    }

    selected_type = st.radio(
        "",
        options=list(part_type_options.keys()),
        format_func=lambda x: part_type_options[x],
        label_visibility="collapsed",
        horizontal=False
    )

    part_color = part_type_colors[selected_type]
    st.markdown(f'<div style="background:#0d0d0d;border:1px solid {part_color}44;border-left:3px solid {part_color};border-radius:8px;padding:0.7rem 1rem;margin-bottom:1rem;font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:{part_color}">Selected: {selected_type} — DFM rules optimized for this process</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Step 2 — Upload your file</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:0.5rem">Supports STL and STEP (.stp / .step) files</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["stl", "step", "stp"], label_visibility="collapsed")

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        file_ext = ".stl" if file_name.endswith(".stl") else ".step" if file_name.endswith(".step") else ".stp"

        with st.spinner("Loading and analyzing geometry..."):
            mesh = load_mesh(uploaded_file, file_ext)

        if mesh is None:
            st.error("Could not load file.")
            st.stop()

        geo = analyze_geometry(mesh)
        stress_data = detect_stress_indicators(mesh, geo)
        issues = run_dfm_checks(geo, thresholds, selected_type)
        score = calculate_score(issues)
        cost_tier, cost_color, cost_bg = get_cost_tier(score)
        cost_data = estimate_cost(geo, score, default_material, selected_type)
        cost_data["material"] = default_material

        # Run simplified FEA immediately
        fea_results = simplified_fea(geo, selected_type, default_material)
        part_context = build_part_context(geo, issues, score, cost_data, selected_type, fea_results, stress_data)

        st.session_state.update({
            "geo": geo, "issues": issues, "score": score,
            "part_type": selected_type, "stress_data": stress_data,
            "fea_results": fea_results, "part_context": part_context,
            "loaded": True, "filename": uploaded_file.name,
            "part_chat": [], "mesh": mesh
        })

        # Auto part summary
        st.markdown('<div class="section-title">AI Part Summary</div>', unsafe_allow_html=True)
        with st.spinner("Generating summary..."):
            summary = generate_part_summary(geo, issues, selected_type)
        st.markdown(f'<div class="summary-box"><div class="ai-label">▸ AUTO ANALYSIS — {uploaded_file.name.upper()} — {selected_type.upper()}</div>{summary}</div>', unsafe_allow_html=True)

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown('<div class="section-title">3D Model Viewer</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#555;margin-bottom:0.5rem">Orange = stress zones · Red = DFM issues · Bright red = critical overlap</div>', unsafe_allow_html=True)

            problem_faces = get_problem_faces(mesh, issues, stress_data)
            stress_faces = stress_data.get("stress_faces", set())
            mesh_json = mesh_to_threejs_json(mesh, problem_faces, stress_faces)
            st.components.v1.html(render_3d_viewer(mesh_json, score), height=460, scrolling=False)

            counts = []
            if problem_faces:
                counts.append(f"⚠ {len(problem_faces)} DFM issue faces (red)")
            if stress_faces:
                counts.append(f"⚡ {len(stress_faces)} stress indicator faces (orange)")
            if not counts:
                counts.append("✓ No problem areas detected")
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#888;margin-top:0.4rem">{" · ".join(counts)}</div>', unsafe_allow_html=True)

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
            pills = [f"Faces: {geo['faces']:,}", f"Aspect: {geo['aspect_ratio']:.1f}:1",
                     f"Surface: {geo['surface_area']:.0f} mm2", f"Watertight: {'Yes' if geo['watertight'] else 'No'}",
                     f"Kt ≈ {stress_data['kt_estimate']}"]
            if geo["fill_ratio"]:
                pills.append(f"Fill: {geo['fill_ratio']:.1f}%")
            pills_html = "".join([f'<span class="stat-pill">{p}</span>' for p in pills])
            st.markdown(f'<div class="stat-row">{pills_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Cost Estimate</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total (INR)</div><div class="metric-value" style="font-size:1.3rem">Rs {cost_data["total_inr"]:,}</div><div class="metric-unit">{default_material}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total (USD)</div><div class="metric-value" style="font-size:1.3rem">${cost_data["total_usd"]}</div><div class="metric-unit">approx</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#555;margin-top:0.5rem">Material: Rs {cost_data["material_inr"]:,} · Process: Rs {cost_data["machining_inr"]:,} · Setup: Rs {cost_data["setup_inr"]:,}</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="section-title">Manufacturability Score</div>', unsafe_allow_html=True)
            score_color_hex = "#38a169" if score >= 85 else "#d69e2e" if score >= 60 else "#e53e3e"
            verdict = "Ready to manufacture" if score >= 85 else "Needs minor revisions" if score >= 60 else "Significant redesign needed" if score >= 35 else "Not manufacturable as-is"
            sc1, sc2 = st.columns([1, 3])
            with sc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Score</div><div class="metric-value" style="color:{score_color_hex};font-size:2.5rem">{score}</div><div class="metric-unit">/ 100</div></div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'<div class="metric-card" style="text-align:left"><div style="color:{score_color_hex};font-size:1rem;font-weight:600;margin-bottom:0.4rem">{verdict}</div><div style="color:#444;font-size:0.8rem;margin-bottom:0.6rem">{len(issues)} issue(s) · {selected_type} · {file_ext.upper()}</div><span style="background:{cost_bg};color:{cost_color};border:1px solid {cost_color}44;padding:0.2rem 0.7rem;border-radius:20px;font-family:IBM Plex Mono,monospace;font-size:0.75rem;font-weight:600">{cost_tier} COST</span></div>', unsafe_allow_html=True)
            st.progress(score / 100)

            st.markdown('<div class="section-title">DFM Issues</div>', unsafe_allow_html=True)
            if not issues:
                st.markdown('<div class="issue-card issue-ok"><div class="issue-title">✓ No issues detected</div><div class="issue-desc">Part meets manufacturability criteria for this process.</div></div>', unsafe_allow_html=True)
            else:
                sev_order = {"critical": 0, "warning": 1, "info": 2}
                for issue in sorted(issues, key=lambda x: sev_order.get(x["severity"], 3)):
                    css = {"critical": "issue-critical", "warning": "issue-warning", "info": "issue-info"}.get(issue["severity"], "issue-info")
                    icon = {"critical": "✕", "warning": "⚠", "info": "ℹ"}.get(issue["severity"], "•")
                    st.markdown(f'<div class="issue-card {css}"><div class="issue-title">{icon} {issue["msg"]}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Stress Indicators</div>', unsafe_allow_html=True)
            si1, si2, si3 = st.columns(3)
            with si1:
                kt = stress_data["kt_estimate"]
                kt_color = "#38a169" if kt < 1.5 else "#d69e2e" if kt < 2.5 else "#e53e3e"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Stress Conc. Kt</div><div class="metric-value" style="color:{kt_color}">{kt}</div><div class="metric-unit">estimate</div></div>', unsafe_allow_html=True)
            with si2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Sharp corners</div><div class="metric-value">{stress_data["sharp_corners"]}</div><div class="metric-unit">detected</div></div>', unsafe_allow_html=True)
            with si3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Thin regions</div><div class="metric-value">{stress_data["thin_sections"]}</div><div class="metric-unit">detected</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">AI Engineering Assessment</div>', unsafe_allow_html=True)
            with st.spinner("Generating assessment..."):
                advice = get_ai_advice(geo, issues, selected_type)
            st.markdown(f'<div class="ai-box"><div class="ai-label">▸ CLAUDE AI — {selected_type.upper()} ENGINEER</div>{advice.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    else:
        st.session_state["loaded"] = False
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.2">⚙</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:#333">Select part type above then upload STL or STEP file</div>
            <div style="font-size:0.8rem;color:#2a2a2a;margin-top:0.5rem">Supports .stl · .step · .stp</div>
        </div>
        """, unsafe_allow_html=True)


# ==================== FEA & STRESS ====================
with tab2:
    st.markdown('<div class="section-title">Simplified Structural Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1rem">Beam bending, deflection, buckling, natural frequency — based on mechanics of materials. Not a full FEA solver but accurate for common shapes.</div>', unsafe_allow_html=True)

    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first in the Analyzer tab</div></div>', unsafe_allow_html=True)
    else:
        geo = st.session_state["geo"]
        part_type = st.session_state["part_type"]

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fea_material = st.selectbox("Material for analysis", [
                "Aluminium 6061", "Mild Steel (1018)", "Stainless Steel 304",
                "Titanium Grade 5", "Brass C360", "Free-machining Steel (12L14)",
                "Tool Steel D2", "Polypropylene", "ABS Plastic", "Nylon PA66"
            ], key="fea_mat")
        with col_f2:
            load_n = st.number_input("Applied load (N)", min_value=1, max_value=100000, value=1000, step=100)

        if st.button("Run Structural Analysis"):
            with st.spinner("Calculating..."):
                fea = simplified_fea(geo, part_type, fea_material, load_n)
            st.session_state["fea_custom"] = fea

        fea = st.session_state.get("fea_custom", st.session_state.get("fea_results"))

        if fea:
            st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                sf_b = fea.get("safety_factor_bending", 0)
                sf_color = "#38a169" if sf_b > 3 else "#d69e2e" if sf_b > 1.5 else "#e53e3e"
                st.markdown(f'<div class="fea-result"><div class="fea-label">Safety factor (bending)</div><div class="fea-value" style="color:{sf_color}">{sf_b}x</div></div>', unsafe_allow_html=True)
            with r2:
                bs = fea.get("bending_stress_mpa", 0)
                ys = fea.get("yield_mpa", 370)
                bs_color = "#38a169" if bs < ys * 0.4 else "#d69e2e" if bs < ys * 0.7 else "#e53e3e"
                st.markdown(f'<div class="fea-result"><div class="fea-label">Max bending stress</div><div class="fea-value" style="color:{bs_color}">{bs} MPa</div></div>', unsafe_allow_html=True)
            with r3:
                defl = fea.get("deflection_mm", 0)
                defl_color = "#38a169" if defl < 1 else "#d69e2e" if defl < 5 else "#e53e3e"
                st.markdown(f'<div class="fea-result"><div class="fea-label">Deflection at {load_n}N</div><div class="fea-value" style="color:{defl_color}">{defl} mm</div></div>', unsafe_allow_html=True)
            with r4:
                mass = fea.get("mass_kg", 0)
                st.markdown(f'<div class="fea-result"><div class="fea-label">Estimated mass</div><div class="fea-value">{mass} kg</div></div>', unsafe_allow_html=True)

            r5, r6, r7, r8 = st.columns(4)
            with r5:
                ax_sf = fea.get("safety_factor_axial", 0)
                ax_color = "#38a169" if ax_sf > 3 else "#d69e2e" if ax_sf > 1.5 else "#e53e3e"
                st.markdown(f'<div class="fea-result"><div class="fea-label">Safety factor (axial)</div><div class="fea-value" style="color:{ax_color}">{ax_sf}x</div></div>', unsafe_allow_html=True)
            with r6:
                ax_s = fea.get("axial_stress_mpa", 0)
                st.markdown(f'<div class="fea-result"><div class="fea-label">Axial stress</div><div class="fea-value">{ax_s} MPa</div></div>', unsafe_allow_html=True)
            with r7:
                buck_sf = fea.get("safety_factor_buckling")
                if buck_sf:
                    buck_color = "#38a169" if buck_sf > 5 else "#d69e2e" if buck_sf > 2 else "#e53e3e"
                    st.markdown(f'<div class="fea-result"><div class="fea-label">Buckling safety factor</div><div class="fea-value" style="color:{buck_color}">{buck_sf}x</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="fea-result"><div class="fea-label">Buckling</div><div class="fea-value" style="color:#38a169">Not critical</div></div>', unsafe_allow_html=True)
            with r8:
                freq = fea.get("natural_freq_hz")
                if freq:
                    st.markdown(f'<div class="fea-result"><div class="fea-label">Natural frequency</div><div class="fea-value">{freq} Hz</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="fea-result"><div class="fea-label">Natural frequency</div><div class="fea-value">N/A</div></div>', unsafe_allow_html=True)

            # Material reference
            st.markdown('<div class="section-title">Material Reference</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#555">Material: {fea_material} · E = {fea.get("E_gpa", 0)} GPa · Yield strength = {fea.get("yield_mpa", 0)} MPa · Load applied = {load_n} N (worst case cantilever)</div>', unsafe_allow_html=True)

            # Stress assessment from AI
            st.markdown('<div class="section-title">AI Stress Assessment</div>', unsafe_allow_html=True)
            stress_data = st.session_state.get("stress_data", {})
            if st.button("Get AI Stress Analysis"):
                with st.spinner("Analyzing stress risks..."):
                    stress_assessment = get_stress_assessment(geo, part_type, stress_data, fea_material)
                st.session_state["stress_assessment"] = stress_assessment

            if "stress_assessment" in st.session_state:
                st.markdown(f'<div class="fea-box"><div class="ai-label">▸ CLAUDE AI — STRESS ENGINEER</div>{st.session_state["stress_assessment"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            st.markdown('<div style="color:#333;font-size:0.75rem;margin-top:1rem">Note: Simplified beam theory. Assumes worst-case cantilever loading. For complex loading or safety-critical parts use full FEA software.</div>', unsafe_allow_html=True)


# ==================== GEOMETRY OPTIMIZER ====================
with tab3:
    st.markdown('<div class="section-title">AI Geometry Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1rem">AI suggests structural geometry changes — ribs, gussets, corrugations, flanges — to increase strength or reduce material cost. Like how corrugated roof sheets are stronger than flat sheets with same material.</div>', unsafe_allow_html=True)

    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first in the Analyzer tab</div></div>', unsafe_allow_html=True)
    else:
        geo = st.session_state["geo"]
        part_type = st.session_state["part_type"]
        issues = st.session_state["issues"]

        opt_goal = st.radio(
            "Optimization goal",
            ["Reduce cost (same strength)", "Increase strength (same material)", "Reduce weight", "Improve machinability"],
            horizontal=True
        )

        if st.button("Generate Geometry Suggestions"):
            with st.spinner("Analyzing geometry and generating optimization suggestions..."):
                opt_prompt = f"""You are a structural engineering and DFM expert.

Part: {geo['length']:.1f} x {geo['width']:.1f} x {geo['height']:.1f} mm
Type: {part_type}
Goal: {opt_goal}
Issues: {', '.join([i['msg'] for i in issues]) if issues else 'none'}

Suggest 5 specific geometry changes to achieve: {opt_goal}

For each suggestion:
CHANGE: [exact geometry change with dimensions]
PRINCIPLE: [structural engineering principle — e.g. "corrugation increases second moment of area I by factor of N"]
BENEFIT: [specific % improvement or mm saving]
HOW TO MAKE IT: [exact manufacturing steps]
EXAMPLE: [real world example of this technique]

Think creatively. Consider:
- Corrugations and ribs for sheet metal stiffness
- I-beam and C-channel profiles for bending resistance
- Gussets and triangulation for joints
- Relief cuts to reduce stress concentration
- Hollow sections with same outer dimensions
- Lattice patterns for weight reduction
- Arc and curved profiles for load distribution

Be very specific with mm values. Reference actual part dimensions."""

                suggestions = safe_ai([{"role": "user", "content": opt_prompt}], max_tokens=800)
            st.session_state["geo_suggestions"] = suggestions

        if "geo_suggestions" in st.session_state:
            st.markdown(f'<div class="geo-box"><div class="ai-label">▸ CLAUDE AI — STRUCTURAL OPTIMIZATION ENGINEER · Goal: {opt_goal.upper()}</div>{st.session_state["geo_suggestions"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


# ==================== MACHINING GUIDE ====================
with tab4:
    st.markdown('<div class="section-title">Manufacturing Guide</div>', unsafe_allow_html=True)
    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first in the Analyzer tab</div></div>', unsafe_allow_html=True)
    else:
        part_type = st.session_state["part_type"]
        guide_mat = st.selectbox("Material", [
            "Aluminium 6061", "Mild Steel (1018)", "Stainless Steel 304",
            "Titanium Grade 5", "Brass C360", "Free-machining Steel (12L14)",
            "Tool Steel D2", "Polypropylene", "ABS Plastic", "Nylon PA66"
        ], key="guide_mat")

        if st.button("Generate Manufacturing Guide"):
            with st.spinner(f"Generating {part_type} guide..."):
                guide = get_machining_guide(st.session_state["geo"], guide_mat, part_type)
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


# ==================== PART CHAT ====================
with tab5:
    st.markdown('<div class="section-title">Part-Aware Engineering Chat</div>', unsafe_allow_html=True)

    if not st.session_state.get("loaded"):
        st.markdown('<div class="issue-card issue-info"><div class="issue-title">ℹ Upload a file first — AI will know your specific part</div></div>', unsafe_allow_html=True)
    else:
        geo = st.session_state["geo"]
        filename = st.session_state.get("filename", "your part")
        part_type = st.session_state["part_type"]
        score = st.session_state["score"]

        st.markdown(f'<div style="background:#0a1a0f;border:1px solid #1a3a1a;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#68d391">✓ AI knows: {filename} · {part_type} · {geo["length"]:.0f}x{geo["width"]:.0f}x{geo["height"]:.0f}mm · Score {score}/100 · Kt={st.session_state["stress_data"].get("kt_estimate", 1.0)}</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#555;font-size:0.85rem;margin-bottom:1rem">Ask anything about your specific part. AI answers based on your actual geometry, stress data, and manufacturing process.</div>', unsafe_allow_html=True)

        if "part_chat" not in st.session_state:
            st.session_state.part_chat = []

        for msg in st.session_state.part_chat:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai"><div class="chat-ai-bubble">{msg["content"].replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)

        if len(st.session_state.part_chat) == 0:
            st.markdown('<div style="color:#333;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem">Suggested questions</div>', unsafe_allow_html=True)
            suggestions = [
                "What material should I use for this part?",
                "Where are the stress concentration points?",
                "How can I reduce the cost of this part?",
                "Is this suitable for the selected process?",
                "What geometry change will make this stronger?",
                "How can I reduce weight without losing strength?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(s, key=f"psug_{i}", use_container_width=True):
                        try:
                            st.session_state.part_chat.append({"role": "user", "content": s})
                            reply = get_context_chat_response(st.session_state.part_chat, st.session_state["part_context"])
                            st.session_state.part_chat.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.session_state.part_chat.append({"role": "assistant", "content": f"Error: {str(e)}"})
                        st.rerun()

        user_input = st.chat_input("Ask about your specific part...")
        if user_input:
            st.session_state.part_chat.append({"role": "user", "content": user_input})
            reply = get_context_chat_response(st.session_state.part_chat, st.session_state["part_context"])
            st.session_state.part_chat.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.part_chat:
            if st.button("Clear chat"):
                st.session_state.part_chat = []
                st.rerun()
