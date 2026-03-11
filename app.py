import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from linkage import FourBarLinkage
from optimizer import GAOptimizer, CylinderCatalogue, FitnessEvaluator

st.set_page_config(layout="wide")
st.title("Linkage Solver: Genetic Optimization")
TOPOLOGY_NAMES = {0: "Lower Arm to Upper Arm", 1: "Frame to Lower Arm", 2: "Frame to Upper Arm"}

with st.sidebar:
    st.header("1. Design Targets")
    target_travel = st.number_input("Target Travel (mm)", value=1000.0)
    load_kg = st.number_input("Load (kg)", value=10000)
    
    st.header("2. Cylinder Specs")
    cyl_diam, nom_press = st.number_input("Cylinder Diameter (in)", value=5.0), st.number_input("Nominal Pressure (psi)", value=2500.0)
    losses, line_fric = st.slider("Efficiency Losses (0-1)", 0.0, 0.5, 0.01), st.slider("Line Friction (0-1)", 0.0, 0.5, 0.15)
    cyl_params = {'cyl_diam_in': cyl_diam, 'nom_press_psi': nom_press, 'losses': losses, 'line_fric': line_fric}

    st.header("3. Hardstop / Clearance")
    arm_width, cyl_env = st.number_input("Arm Width (mm)", value=100.0), st.number_input("Cylinder Envelope (mm)", value=152.4)
    use_clearance = st.checkbox("Enforce Clearance", value=True)

    st.header("4. Geometry Constraints")
    fixed_params = {}
    def lockable_input(label, key, default_val):
        col1, col2 = st.columns([2, 1])
        val = col1.number_input(label, value=float(default_val))
        if col2.checkbox("Hold", key=f"lock_{key}"): fixed_params[key] = int(val)
        return val
    lockable_input("Arm Lower (L_L)", "L_L", 800); lockable_input("Arm Upper (L_U)", "L_U", 800)
    lockable_input("Frame H (H_f)", "H_f", 350); lockable_input("Effector H (H_e)", "H_e", 350)
    
    st.write("**Allowed Topologies**")
    allowed_topos = [k for k, v in TOPOLOGY_NAMES.items() if st.checkbox(v, value=True, key=f"topo_{k}")]
    if not allowed_topos: allowed_topos = [0]
    symmetrical_lugs = st.checkbox("Symmetrical Lugs (LU only)", value=True)
    
    st.header("5. Optimization Settings")
    generations = st.slider("Generations", 10, 1000, 100)
    if st.button("🚀 Run Optimizer"):
        with st.spinner("Evolving solutions..."):
            opt = GAOptimizer(target_travel, load_kg, cyl_params, fixed_params, allowed_topos, symmetrical_lugs, arm_width, cyl_env, use_clearance)
            st.session_state.top_solutions = opt.run(gens=generations)
            st.session_state.run_symmetrical = symmetrical_lugs

if 'top_solutions' in st.session_state:
    st.header("Results Comparison")
    cols_top = st.columns(len(st.session_state.top_solutions))
    for i, (sol, fit) in enumerate(st.session_state.top_solutions):
        if cols_top[i].button(f"Solution {i+1}\nFit: {int(fit)}"): st.session_state.selected_idx = i
    current_idx = st.session_state.get('selected_idx', 0)
    sol, _ = st.session_state.top_solutions[current_idx]
    
    L_L, L_U, H_f, H_e, dx_f = int(sol[0]), int(sol[1]), int(sol[2]), int(sol[3]), int(sol[4])
    topo, stroke_idx = int(sol[5]), int(sol[10])
    
    st.subheader("Manual Tweaking")
    cols = st.columns(4)
    with cols[0]: L_L = st.number_input("L_Lower", value=L_L); L_U = st.number_input("L_Upper", value=L_U)
    with cols[1]: H_f = st.number_input("H_Frame", value=H_f); H_e = st.number_input("H_Effector", value=H_e)
    with cols[2]: dx_f = st.number_input("dx_Frame", value=dx_f); topo = st.selectbox("Topology", list(TOPOLOGY_NAMES.keys()), index=topo, format_func=lambda x: TOPOLOGY_NAMES[x])
    with cols[3]: stroke_idx = st.selectbox("Cylinder Stroke", range(len(CylinderCatalogue.STROKES_IN)), index=stroke_idx, format_func=lambda x: f"{CylinderCatalogue.STROKES_IN[x]} in")

    st.subheader("Lug Positions")
    cols2 = st.columns(4)
    is_sym = (topo == 0 and st.session_state.get('run_symmetrical', False))
    with cols2[0]: u_pct = st.slider("Lug1 U %", 0, 100, int(sol[6]/10)) / 100.0
    with cols2[1]: v_val = st.slider("Lug1 V (mm)", -300, 300, int(((sol[7]/1000.0)*600)-300))
    if is_sym:
        u1, v1, u2, v2 = u_pct * L_L, v_val, (1 - u_pct) * L_U, -v_val
        with cols2[2]: st.write(f"**Lug2 U % (Sym):** {(1-u_pct)*100:.1f}%")
        with cols2[3]: st.write(f"**Lug2 V (Sym):** {-v_val:.1f} mm")
    else:
        with cols2[2]: u2_pct = st.slider("Lug2 U %", 0, 100, int(sol[8]/10)) / 100.0
        with cols2[3]: v2_val = st.slider("Lug2 V (mm)", -300, 300, int(((sol[9]/1000.0)*600)-300))
        l2_arm = L_L if topo == 1 else L_U
        u1, v1, u2, v2 = u_pct * L_L, v_val, u2_pct * l2_arm, v2_val

    linkage = FourBarLinkage(L_L, L_U, H_f, H_e, dx_f)
    if topo == 0: lug1, lug2 = ('L', u1, v1), ('U', u2, v2)
    elif topo == 1: lug1, lug2 = ('F', ((sol[6]/1000.0)*1500)-750, ((sol[7]/1000.0)*1500)-750), ('L', u2, v2)
    else: lug1, lug2 = ('F', ((sol[6]/1000.0)*1500)-750, ((sol[7]/1000.0)*1500)-750), ('U', u2, v2)
        
    try:
        res = linkage.analyze_range(-45, 45, lug1, lug2)
        evaluator = FitnessEvaluator(target_travel, load_kg, cyl_params, arm_width=arm_width, cyl_env=cyl_env)
        req_force = ((load_kg * 9.81) * res['mech_ratio']) / (1000 * 9.81)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        import matplotlib.patches as patches
        def plot_frame(idx, color, alpha):
            half_arm = arm_width / 2.0
            ax1.add_patch(patches.Rectangle((0, -half_arm), L_L, arm_width, angle=np.degrees(res['thetas'][idx]), rotation_point=(0,0), edgecolor=color, facecolor='none', alpha=alpha*0.3))
            ax1.add_patch(patches.Rectangle((linkage.P2[0], linkage.P2[1] - half_arm), L_U, arm_width, angle=np.degrees(res['phis'][idx]), rotation_point=(linkage.P2[0], linkage.P2[1]), edgecolor=color, facecolor='none', alpha=alpha*0.3))
            q1, q2 = res['Q1'][:, idx], res['Q2'][:, idx]
            ax1.add_patch(patches.Rectangle((q1[0], q1[1] - cyl_env/2.0), np.linalg.norm(q2-q1), cyl_env, angle=np.degrees(np.arctan2(q2[1]-q1[1], q2[0]-q1[0])), rotation_point=(q1[0], q1[1]), edgecolor='red', facecolor='red', alpha=alpha*0.1))
            ax1.plot([0, linkage.P2[0]], [0, linkage.P2[1]], 'k-o', alpha=alpha, lw=2)
            ax1.plot([0, res['P3'][0, idx]], [0, res['P3'][1, idx]], '--', color=color, alpha=alpha)
            ax1.plot([linkage.P2[0], res['P4'][0, idx]], [linkage.P2[1], res['P4'][1, idx]], '--', color=color, alpha=alpha)
            ax1.plot([res['P3'][0, idx], res['P4'][0, idx]], [res['P3'][1, idx], res['P4'][1, idx]], 'k-o', alpha=alpha, lw=3)
            ax1.plot([q1[0], q2[0]], [q1[1], q2[1]], 'r-o', alpha=alpha, lw=2)
        plot_frame(0, 'gray', 0.3); plot_frame(-1, 'blue', 1.0)
        ax1.set_aspect('equal'); ax1.set_title("Linkage Wireframe")
        travel = res['y_effector'] - res['y_effector'][0]
        ax2.plot(travel, req_force, label="Req. Force (T)")
        ax2.axhline(evaluator.cyl_capacity_tonnes, color='r', ls='--', label="Capacity")
        ax2.set_xlabel("Travel (mm)"); ax2.set_ylabel("Force (Tonnes)")
        ax2.set_ylim(0, min(20000, max(evaluator.cyl_capacity_tonnes * 2, np.max(req_force) * 1.1)))
        ax2.legend(); ax2.set_title("Force Profile")
        st.pyplot(fig)
        st.write(f"**Travel:** {travel[-1]:.1f} mm | **Peak Force:** {np.max(req_force):.2f} T | **Cyl Stroke Req:** {(np.max(res['l_cyl']) - np.min(res['l_cyl']))/25.4:.2f} in | **Validity:** {res['valid_fraction']*100:.1f}%")
    except Exception as e: st.error(f"Analysis failed: {e}")
