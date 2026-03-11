import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from linkage import FourBarLinkage
from optimizer import GAOptimizer, CylinderCatalogue, FitnessEvaluator

st.set_page_config(layout="wide", page_title="Linkage Solver")
TOPOLOGY_NAMES = {0: "Lower ↔ Upper", 1: "Frame → Lower", 2: "Frame → Upper"}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Wider sidebar — initial width, resizable */
    section[data-testid="stSidebar"] { min-width: 440px; }
    section[data-testid="stSidebar"] > div { min-width: 440px; }

    /* Global typography */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; }

    /* Tighten main header */
    .block-container { padding-top: 1.5rem !important; }
    h1 { font-size: 1.4rem !important; margin-bottom: 0.25rem !important; letter-spacing: -0.02em; }

    /* Sidebar section labels — smaller, uppercase, spaced */
    section[data-testid="stSidebar"] h2 {
        font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.1em;
        color: #888 !important; margin: 1.2rem 0 0.3rem 0 !important; padding: 0 !important;
        border-bottom: 1px solid #333; padding-bottom: 0.3rem !important;
    }

    /* Main area h2 — medium, no excess space */
    .main h2 { font-size: 1rem !important; margin: 0.8rem 0 0.4rem 0 !important; letter-spacing: -0.01em; }
    .main h3 { font-size: 0.85rem !important; margin: 0.5rem 0 0.3rem 0 !important; }

    /* Compact inputs */
    .stNumberInput label, .stSlider label, .stSelectbox label, .stCheckbox label {
        font-size: 0.78rem !important; font-weight: 500 !important; color: #aaa !important;
    }
    .stNumberInput input { font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; }

    /* Solution buttons */
    div[data-testid="stHorizontalBlock"] button {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important; white-space: pre-line;
    }

    /* Metrics row */
    .metric-row {
        display: flex; gap: 1.5rem; padding: 0.5rem 0; flex-wrap: wrap;
        font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    }
    .metric-row .m-label { color: #888; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-row .m-val { font-weight: 500; }
    .metric-row .m-ok { color: #4caf50; }
    .metric-row .m-warn { color: #ff9800; }
    .metric-row .m-bad { color: #f44336; }

    /* Hide fullscreen button on charts */
    button[title="View fullscreen"] { display: none !important; }

    /* Tighter expander */
    .streamlit-expanderHeader { font-size: 0.82rem !important; font-weight: 500 !important; }
</style>""", unsafe_allow_html=True)

st.markdown("# Linkage Solver")

# ── Helpers ───────────────────────────────────────────────────────────────────
def dual_input(label, min_v, max_v, default, key, step=1, show_label=True):
    """Slider + text entry side by side, integer-only, bidirectionally synced."""
    sk, nk = f"{key}_s", f"{key}_n"
    if sk not in st.session_state and nk not in st.session_state:
        st.session_state[sk] = int(default)
        st.session_state[nk] = int(default)
    def _on_slider(): st.session_state[nk] = st.session_state[sk]
    def _on_number(): st.session_state[sk] = st.session_state[nk]
    if show_label:
        st.markdown(f'<p style="font-size:0.78rem;font-weight:500;color:#aaa;margin:0 0 0.2rem 0">{label}</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    c1.slider(label, min_value=int(min_v), max_value=int(max_v), step=int(step), key=sk, on_change=_on_slider, label_visibility="collapsed")
    c2.number_input(label, min_value=int(min_v), max_value=int(max_v), step=int(step), key=nk, on_change=_on_number, label_visibility="collapsed")
    return int(st.session_state[sk])

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Design Targets")
    target_travel = dual_input("Target Travel (mm)", 100, 5000, 1000, "target_travel", step=50)
    load_kg = dual_input("Load (kg)", 100, 200000, 10000, "load_kg", step=500)

    st.header("Cylinder")
    c1, c2 = st.columns(2)
    cyl_diam = c1.number_input("Bore (in)", value=5.0, step=0.5, key="cyl_d", min_value=1.0, max_value=20.0)
    nom_press = c2.number_input("Pressure (psi)", value=2500, step=100, key="cyl_p", min_value=500, max_value=10000)
    c3, c4 = st.columns(2)
    losses = c3.slider("Losses", 0.0, 0.5, 0.01, key="losses")
    line_fric = c4.slider("Line Friction", 0.0, 0.5, 0.15, key="fric")
    cyl_params = {'cyl_diam_in': cyl_diam, 'nom_press_psi': nom_press, 'losses': losses, 'line_fric': line_fric}

    st.header("Clearance")
    arm_width = dual_input("Arm Width (mm)", 20, 500, 100, "aw", step=10)
    cyl_env = dual_input("Cylinder Envelope (mm)", 50, 500, 152, "ce", step=10)
    use_clearance = st.checkbox("Enforce Clearance", value=True)

    st.header("Geometry Holds")
    fixed_params = {}
    def lockable(label, key, default_val, min_v, max_v, step=10):
        sk, nk = f"geo_{key}_s", f"geo_{key}_n"
        if sk not in st.session_state and nk not in st.session_state:
            st.session_state[sk] = int(default_val)
            st.session_state[nk] = int(default_val)
        def _on_s(): st.session_state[nk] = st.session_state[sk]
        def _on_n(): st.session_state[sk] = st.session_state[nk]
        st.markdown(f'<p style="font-size:0.78rem;font-weight:500;color:#aaa;margin:0 0 0.2rem 0">{label}</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([5, 2, 1.5])
        c1.slider(label, min_value=int(min_v), max_value=int(max_v), step=int(step), key=sk, on_change=_on_s, label_visibility="collapsed")
        c2.number_input(label, min_value=int(min_v), max_value=int(max_v), step=int(step), key=nk, on_change=_on_n, label_visibility="collapsed")
        val = int(st.session_state[sk])
        if c3.checkbox("Hold", key=f"lock_{key}"):
            fixed_params[key] = val
        return val
    lockable("L Lower", "L_L", 800, 200, 2000)
    lockable("L Upper", "L_U", 800, 200, 2000)
    lockable("H Frame", "H_f", 350, 50, 1000)
    lockable("H Effector", "H_e", 350, 50, 1000)
    lockable("dx Frame", "dx_f", 0, -500, 500)

    st.header("Topology")
    allowed_topos = [k for k, v in TOPOLOGY_NAMES.items() if st.checkbox(v, value=True, key=f"topo_{k}")]
    if not allowed_topos: allowed_topos = [0]
    symmetrical_lugs = st.checkbox("Symmetrical Lugs (LU only)", value=True)

    st.header("Optimizer")
    stroke_names = [f"{s} in" for s in CylinderCatalogue.STROKES_IN]
    use_stroke_pref = st.checkbox("Prefer Stroke", value=True)
    preferred_stroke_idx = None
    if use_stroke_pref:
        preferred_stroke_idx = st.selectbox("Preferred Stroke", range(len(stroke_names)), index=2, format_func=lambda i: stroke_names[i])
    generations = dual_input("Generations", 10, 1000, 100, "gens", step=10)
    if st.button("Run Optimizer", use_container_width=True, type="primary"):
        with st.spinner("Evolving..."):
            opt = GAOptimizer(target_travel, load_kg, cyl_params, fixed_params, allowed_topos, symmetrical_lugs, arm_width, cyl_env, use_clearance, preferred_stroke_idx)
            st.session_state.top_solutions = opt.run(gens=generations)

# ── GA Results ────────────────────────────────────────────────────────────────
if 'top_solutions' in st.session_state:
    st.markdown("## Solutions")
    if not st.session_state.top_solutions:
        st.warning("No feasible solutions. Relax constraints or increase generations.")
    else:
        cols_top = st.columns(len(st.session_state.top_solutions))
        for i, (sol, fit) in enumerate(st.session_state.top_solutions):
            _, _, _, _, _, _, _, s_idx = FitnessEvaluator(target_travel, load_kg, cyl_params).decode_genome(sol)
            stroke_in = CylinderCatalogue.STROKES_IN[s_idx]
            if cols_top[i].button(f"#{i+1}  fit {int(fit)}  {stroke_in}in", key=f"sol_{i}", use_container_width=True):
                st.session_state.selected_idx = i

# ── Linkage Configuration ─────────────────────────────────────────────────────
st.markdown("## Configuration")

def _ga_sol():
    if 'top_solutions' not in st.session_state or not st.session_state.top_solutions:
        return None
    idx = st.session_state.get('selected_idx', 0)
    if idx >= len(st.session_state.top_solutions):
        idx = 0
    return st.session_state.top_solutions[idx][0]

sol = _ga_sol()

c1, c2, c3, c4 = st.columns(4)
with c1:
    L_L = dual_input("L Lower (mm)", 100, 2000, int(sol[0]) if sol is not None else 800, "cfg_LL", step=10)
    L_U = dual_input("L Upper (mm)", 100, 2000, int(sol[1]) if sol is not None else 800, "cfg_LU", step=10)
with c2:
    H_f = dual_input("H Frame (mm)", 50, 1000, int(sol[2]) if sol is not None else 350, "cfg_Hf", step=10)
    H_e = dual_input("H Effector (mm)", 50, 1000, int(sol[3]) if sol is not None else 350, "cfg_He", step=10)
with c3:
    dx_f = dual_input("dx Frame (mm)", -500, 500, int(sol[4]) if sol is not None else 0, "cfg_dxf", step=10)
    topo = st.selectbox("Topology", list(TOPOLOGY_NAMES.keys()),
                        index=int(sol[5]) if sol is not None else 0,
                        format_func=lambda x: TOPOLOGY_NAMES[x], key="cfg_topo")
with c4:
    stroke_idx = st.selectbox("Cylinder Stroke", range(len(CylinderCatalogue.STROKES_IN)),
                              index=int(sol[10]) if sol is not None else 2,
                              format_func=lambda x: f"{CylinderCatalogue.STROKES_IN[x]} in", key="cfg_stroke")

# ── Lug positions ─────────────────────────────────────────────────────────────
st.markdown("### Lugs")
cols2 = st.columns(4)
is_sym = (topo == 0 and symmetrical_lugs)

if topo == 0:
    default_u = int(sol[6] / 10) if sol is not None else 20
    default_v = int(((sol[7] / 1000.0) * 600) - 300) if sol is not None else 150
    with cols2[0]: u_pct = dual_input("Lug1 U %", 0, 100, default_u, "l1u") / 100.0
    with cols2[1]: v_val = dual_input("Lug1 V (mm)", -300, 300, default_v, "l1v")
    if is_sym:
        u1, v1, u2, v2 = u_pct * L_L, v_val, (1 - u_pct) * L_U, -v_val
        with cols2[2]: st.markdown(f"**Lug2 U** {(1-u_pct)*100:.0f}%")
        with cols2[3]: st.markdown(f"**Lug2 V** {-v_val} mm")
    else:
        default_u2 = int(sol[8] / 10) if sol is not None else 80
        default_v2 = int(((sol[9] / 1000.0) * 600) - 300) if sol is not None else -150
        with cols2[2]: u2_pct = dual_input("Lug2 U %", 0, 100, default_u2, "l2u") / 100.0
        with cols2[3]: v2_val = dual_input("Lug2 V (mm)", -300, 300, default_v2, "l2v")
        u1, v1, u2, v2 = u_pct * L_L, v_val, u2_pct * L_U, v2_val
    lug1, lug2 = ('L', u1, v1), ('U', u2, v2)
else:
    default_fx = int(((sol[6] / 1000.0) * 1500) - 750) if sol is not None else 0
    default_fy = int(((sol[7] / 1000.0) * 1500) - 750) if sol is not None else -200
    default_u2 = int(sol[8] / 10) if sol is not None else 30
    default_v2 = int(((sol[9] / 1000.0) * 600) - 300) if sol is not None else 150
    with cols2[0]: fx = dual_input("Frame Lug X (mm)", -750, 750, default_fx, "flx")
    with cols2[1]: fy = dual_input("Frame Lug Y (mm)", -750, 750, default_fy, "fly")
    with cols2[2]: u2_pct = dual_input("Arm Lug U %", 0, 100, default_u2, "alu") / 100.0
    with cols2[3]: v2_val = dual_input("Arm Lug V (mm)", -300, 300, default_v2, "alv")
    l2_arm = L_L if topo == 1 else L_U
    l2_member = 'L' if topo == 1 else 'U'
    lug1 = ('F', fx, fy)
    lug2 = (l2_member, u2_pct * l2_arm, v2_val)

# ── Analysis & Plots ──────────────────────────────────────────────────────────
linkage = FourBarLinkage(L_L, L_U, H_f, H_e, dx_f)
try:
    res = linkage.analyze_range(-45, 45, lug1, lug2)
    evaluator = FitnessEvaluator(target_travel, load_kg, cyl_params, arm_width=arm_width, cyl_env=cyl_env)
    req_force = ((load_kg * 9.81) * res['mech_ratio']) / (1000 * 9.81)
    min_spec, max_spec, _ = CylinderCatalogue.get_specs(stroke_idx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='#888'); ax.xaxis.label.set_color('#aaa'); ax.yaxis.label.set_color('#aaa')
        ax.title.set_color('#ddd')
        for spine in ax.spines.values(): spine.set_color('#333')

    def plot_frame(idx, color, alpha):
        half_arm = arm_width / 2.0
        ax1.add_patch(patches.Rectangle((0, -half_arm), L_L, arm_width, angle=np.degrees(res['thetas'][idx]), rotation_point=(0,0), edgecolor=color, facecolor='none', alpha=alpha*0.3, lw=1.5))
        ax1.add_patch(patches.Rectangle((linkage.P2[0], linkage.P2[1] - half_arm), L_U, arm_width, angle=np.degrees(res['phis'][idx]), rotation_point=(linkage.P2[0], linkage.P2[1]), edgecolor=color, facecolor='none', alpha=alpha*0.3, lw=1.5))
        q1, q2 = res['Q1'][:, idx], res['Q2'][:, idx]
        ax1.add_patch(patches.Rectangle((q1[0], q1[1] - cyl_env/2.0), np.linalg.norm(q2-q1), cyl_env, angle=np.degrees(np.arctan2(q2[1]-q1[1], q2[0]-q1[0])), rotation_point=(q1[0], q1[1]), edgecolor='#e74c3c', facecolor='#e74c3c', alpha=alpha*0.1))
        ax1.plot([0, linkage.P2[0]], [0, linkage.P2[1]], '-o', color='#ccc', alpha=alpha, lw=2, markersize=4)
        ax1.plot([0, res['P3'][0, idx]], [0, res['P3'][1, idx]], '--', color=color, alpha=alpha, lw=1)
        ax1.plot([linkage.P2[0], res['P4'][0, idx]], [linkage.P2[1], res['P4'][1, idx]], '--', color=color, alpha=alpha, lw=1)
        ax1.plot([res['P3'][0, idx], res['P4'][0, idx]], [res['P3'][1, idx], res['P4'][1, idx]], '-o', color='#ccc', alpha=alpha, lw=2.5, markersize=4)
        ax1.plot([q1[0], q2[0]], [q1[1], q2[1]], '-o', color='#e74c3c', alpha=alpha, lw=2, markersize=3)
    plot_frame(0, '#666', 0.3); plot_frame(-1, '#5dade2', 1.0)
    ax1.set_aspect('equal'); ax1.set_title("Linkage Geometry", fontsize=11)
    ax1.grid(True, alpha=0.1, color='#555')

    travel = res['y_effector'] - res['y_effector'][0]
    ax2.plot(travel, req_force, color='#5dade2', lw=2, label="Required Force")
    ax2.axhline(evaluator.cyl_capacity_tonnes, color='#e74c3c', ls='--', lw=1.5, label="Capacity", alpha=0.8)
    ax2.set_xlabel("Travel (mm)", fontsize=10); ax2.set_ylabel("Force (T)", fontsize=10)
    ax2.set_ylim(0, min(20000, max(evaluator.cyl_capacity_tonnes * 2, np.max(req_force) * 1.1)))
    ax2.legend(fontsize=9, facecolor='#0e1117', edgecolor='#333', labelcolor='#aaa')
    ax2.set_title("Force Profile", fontsize=11)
    ax2.grid(True, alpha=0.1, color='#555')
    # Metrics row (above plots)
    cyl_stroke_req = (np.max(res['l_cyl']) - np.min(res['l_cyl'])) / 25.4
    cyl_min, cyl_max = np.min(res['l_cyl']), np.max(res['l_cyl'])
    peak_force = np.max(req_force)
    in_spec = cyl_min >= min_spec and cyl_max <= max_spec
    under_cap = peak_force <= evaluator.cyl_capacity_tonnes
    spec_cls = "m-ok" if in_spec else "m-bad"
    force_cls = "m-ok" if under_cap else "m-bad"

    st.markdown(f"""<div class="metric-row" style="justify-content:center;">
        <div><span class="m-label">Travel</span><br><span class="m-val">{travel[-1]:.0f} mm</span></div>
        <div><span class="m-label">Peak Force</span><br><span class="m-val {force_cls}">{peak_force:.2f} T</span></div>
        <div><span class="m-label">Capacity</span><br><span class="m-val">{evaluator.cyl_capacity_tonnes:.2f} T</span></div>
        <div><span class="m-label">Cyl Range</span><br><span class="m-val {spec_cls}">{cyl_min:.0f}–{cyl_max:.0f} mm</span></div>
        <div><span class="m-label">Spec Window</span><br><span class="m-val">{min_spec:.0f}–{max_spec:.0f} mm</span></div>
        <div><span class="m-label">Stroke Used</span><br><span class="m-val">{cyl_stroke_req:.1f} in</span></div>
        <div><span class="m-label">Validity</span><br><span class="m-val">{res['valid_fraction']*100:.0f}%</span></div>
    </div>""", unsafe_allow_html=True)

    fig.tight_layout(pad=2)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Analysis failed: {e}")
