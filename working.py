import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy.optimize import minimize

def segment_distance(p1, p2, p3, p4):
    u = p2 - p1; v = p4 - p3; w = p1 - p3
    a = np.sum(u*u, axis=0); b = np.sum(u*v, axis=0); c = np.sum(v*v, axis=0)
    d = np.sum(u*w, axis=0); e = np.sum(v*w, axis=0); D = a*c - b*b
    sN = np.zeros_like(D); sD = D.copy(); tN = np.zeros_like(D); tD = D.copy()
    
    small = D < 1e-8
    sN[small] = 0.0; sD[small] = 1.0; tN[small] = e[small]; tD[small] = c[small]
    ok = ~small
    sN[ok] = b[ok]*e[ok] - c[ok]*d[ok]; tN[ok] = a[ok]*e[ok] - b[ok]*d[ok]
    
    s_lt_0 = ok & (sN < 0.0)
    sN[s_lt_0] = 0.0; tN[s_lt_0] = e[s_lt_0]; tD[s_lt_0] = c[s_lt_0]
    s_gt_1 = ok & (sN > sD)
    sN[s_gt_1] = sD[s_gt_1]; tN[s_gt_1] = e[s_gt_1] + b[s_gt_1]; tD[s_gt_1] = c[s_gt_1]
    
    t_lt_0 = tN < 0.0
    tN[t_lt_0] = 0.0; sN[t_lt_0] = np.clip(-d[t_lt_0], 0.0, a[t_lt_0]); sD[t_lt_0] = a[t_lt_0]
    t_gt_1 = tN > tD
    tN[t_gt_1] = tD[t_gt_1]; sN[t_gt_1] = np.clip((-d[t_gt_1] + b[t_gt_1]), 0.0, a[t_gt_1]); sD[t_gt_1] = a[t_gt_1]
    
    sc = np.where(np.abs(sN) < 1e-8, 0.0, sN / sD)
    tc = np.where(np.abs(tN) < 1e-8, 0.0, tN / tD)
    
    dP = w + (sc * u) - (tc * v)
    return np.sqrt(np.sum(dP*dP, axis=0))


# --- 1. Define UI Widgets ---
style = {'description_width': 'initial'}
layout = widgets.Layout(width='350px')

# Helper for generating slider+checkbox pairs
def make_param(val, min_v, max_v, step_v, desc, is_int=False):
    slider_cls = widgets.IntSlider if is_int else widgets.FloatSlider
    slider = slider_cls(value=val, min=min_v, max=max_v, step=step_v, description=desc, style=style, layout=layout)
    check = widgets.Checkbox(value=False, description='Hold', indent=False, layout=widgets.Layout(width='auto'))
    return slider, check

w_L, c_L = make_param(795.0, 200.0, 2500.0, 1.0, 'Arm Length (L)')
w_yf, c_yf = make_param(350.0, 100.0, 800.0, 1.0, 'Frame y_f')
w_load = widgets.IntSlider(value=10000, min=1000, max=20000, step=100, description='Load (kg)', style=style, layout=layout) # load is never optimized

w_dx_L, c_dx_L = make_param(25.0, 0.0, 1500.0, 1.0, 'Lower Lug ∥ arm (dx_L)')
w_dy_L, c_dy_L = make_param(150.0, -300.0, 300.0, 1.0, 'Lower Lug ⊥ arm (dy_L)')
w_dx_U, c_dx_U = make_param(25.0, 0.0, 1500.0, 1.0, 'Upper Lug ∥ arm (dx_U)')
w_dy_U, c_dy_U = make_param(-150.0, -300.0, 300.0, 1.0, 'Upper Lug ⊥ arm (dy_U)')

w_theta_start, c_theta_start = make_param(-30.0, -60.0, 0.0, 0.5, 'Start Angle (deg)')
w_theta_end, c_theta_end = make_param(20.0, 0.0, 60.0, 0.5, 'End Angle (deg)')

# Engineering Verification Fields
w_cyl_diam = widgets.FloatText(value=5.0, description='Cyl Dia (in):', style=style, layout=widgets.Layout(width='160px'))
w_nom_press = widgets.FloatText(value=2500.0, description='Nom Press (psi):', style=style, layout=widgets.Layout(width='160px'))
w_losses = widgets.FloatText(value=0.01, description='Losses (0-1):', style=style, layout=widgets.Layout(width='160px'))
w_line_fric = widgets.FloatText(value=0.15, description='Line Fric (0-1):', style=style, layout=widgets.Layout(width='160px'))

# Optimizer Controls
w_target_dy = widgets.FloatText(value=1000.0, description='Target dY (mm):', style=style, layout=widgets.Layout(width='200px'))
c_target_dy = widgets.Checkbox(value=False, description='Hold', indent=False, layout=widgets.Layout(width='auto'))
btn_optimize = widgets.Button(description="Auto-Optimize Lugs", button_style='success', icon='cogs')
out_status = widgets.Output()

# --- 2. The Plotting Engine (Triggered by Sliders) ---
def update_dashboard(L, y_f, dx_L, dy_L, dx_U, dy_U, theta_start, theta_end, load_kg, cyl_diam, nom_press, losses, line_fric):
    xc_L = L - dx_L
    yc_L = dy_L
    xc_U = dx_U
    yc_U = dy_U
    theta_rad = np.linspace(np.radians(theta_start), np.radians(theta_end), 500)
    y_wheel = L * np.sin(theta_rad)
    
    x_lower = (xc_L * np.cos(theta_rad)) - (yc_L * np.sin(theta_rad))
    y_lower = (xc_L * np.sin(theta_rad)) + (yc_L * np.cos(theta_rad))
    x_upper = (xc_U * np.cos(theta_rad)) - (yc_U * np.sin(theta_rad))
    y_upper = y_f + (xc_U * np.sin(theta_rad)) + (yc_U * np.cos(theta_rad))
    
    l_cyl = np.sqrt((x_upper - x_lower)**2 + (y_upper - y_lower)**2)
    dy_wheel = np.diff(y_wheel)
    dl_cyl = np.diff(l_cyl)
    dl_cyl[dl_cyl == 0] = 1e-6 
    
    mech_ratio = np.abs(dy_wheel / dl_cyl)
    f_cyl_req_tonnes = ((load_kg * 9.81) * mech_ratio) / (1000 * 9.81)
    travel_mm = y_wheel[:-1] - y_wheel[0]
    
    # Engineering Capacity Calculation
    # Re-interpreting user's new defaults (0.01/0.15) as percentage *reductions*
    final_pressure_psi = nom_press * (1.0 - losses) * (1.0 - line_fric)
    piston_area_in2 = np.pi * (cyl_diam / 2.0)**2
    cyl_force_lbf = final_pressure_psi * piston_area_in2
    # 1 lbf = 4.44822 N
    cyl_force_n = cyl_force_lbf * 4.44822
    cyl_capacity_tonnes = cyl_force_n / (1000 * 9.81)

    # Render Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wireframe
    ax1.set_title("Linkage Wireframe")
    ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)")
    ax1.grid(True, linestyle='--', alpha=0.5); ax1.axis('equal')
    
    import matplotlib.patches as patches
    
    def draw_state(idx, color, alpha_val):
        ax1.plot([0, 0], [0, y_f], 'ks-', markersize=8) 
        
        # Arm rectangles (100mm wide)
        arm_w = 100.0 / 2.0
        deg = np.degrees(theta_rad[idx])
        
        rect_L = patches.Rectangle((0, -arm_w), L, 2*arm_w, angle=deg, rotation_point=(0,0), 
                                   linewidth=1, edgecolor=color, facecolor='none', alpha=alpha_val)
        ax1.add_patch(rect_L)
        
        rect_U = patches.Rectangle((0, y_f - arm_w), L, 2*arm_w, angle=deg, rotation_point=(0, y_f), 
                                   linewidth=1, edgecolor=color, facecolor='none', alpha=alpha_val)
        ax1.add_patch(rect_U)
        
        # Cylinder rectangle (152.4mm wide)
        cyl_w = 152.4 / 2.0
        dx = x_upper[idx] - x_lower[idx]
        dy = y_upper[idx] - y_lower[idx]
        cyl_len = np.sqrt(dx**2 + dy**2)
        cyl_deg = np.degrees(np.arctan2(dy, dx))
        rect_cyl = patches.Rectangle((x_lower[idx], y_lower[idx] - cyl_w), cyl_len, 2*cyl_w, angle=cyl_deg, 
                                     rotation_point=(x_lower[idx], y_lower[idx]), 
                                     linewidth=1, edgecolor='red', facecolor='red', alpha=alpha_val*0.2)
        ax1.add_patch(rect_cyl)
        
        # Centerlines
        ax1.plot([0, L*np.cos(theta_rad[idx])], [0, L*np.sin(theta_rad[idx])], color=color, lw=1, alpha=alpha_val, ls='--')
        ax1.plot([0, L*np.cos(theta_rad[idx])], [y_f, y_f + L*np.sin(theta_rad[idx])], color=color, lw=1, alpha=alpha_val, ls='--')
        ax1.plot([L*np.cos(theta_rad[idx]), L*np.cos(theta_rad[idx])], 
                 [L*np.sin(theta_rad[idx]), y_f + L*np.sin(theta_rad[idx])], 'k-', lw=5, alpha=alpha_val)
        ax1.plot([x_lower[idx], x_upper[idx]], [y_lower[idx], y_upper[idx]], 'r-o', lw=2, alpha=alpha_val)

    draw_state(0, 'gray', 0.3)
    draw_state(-1, 'blue', 1.0)
    
    # Force Curve
    ax2.set_title("Required Cylinder Force vs. Travel")
    ax2.set_xlabel("Travel (mm)"); ax2.set_ylabel("Force (Tonnes)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Shade background based on capacity
    is_safe = np.all(f_cyl_req_tonnes <= cyl_capacity_tonnes)
    if is_safe:
        ax2.set_facecolor('#eaffea') # Light green
    else:
        ax2.set_facecolor('#ffeaea') # Light red
        
    ax2.plot(travel_mm, f_cyl_req_tonnes, 'b-', lw=3, label='Req. Force')
    ax2.plot(travel_mm[-1], f_cyl_req_tonnes[-1], 'ro', markersize=8)
    
    # Let y-axis auto-scale just to the required force
    ax2.set_ylim(bottom=np.min(f_cyl_req_tonnes) * 0.9, top=np.max(f_cyl_req_tonnes) * 1.1)
    
    # Live Readouts
    cyl_min_in = np.min(l_cyl)/25.4
    cyl_max_in = np.max(l_cyl)/25.4
    stats = (f"Travel: {travel_mm[-1]:.1f} mm\n"
             f"Cyl Stroke Req: {cyl_max_in - cyl_min_in:.2f} in\n"
             f"Min Pin-to-Pin: {cyl_min_in:.1f} in\n"
             f"Peak Force (Lifted): {f_cyl_req_tonnes[-1]:.2f} T\n"
             f"Capacity Margin: {cyl_capacity_tonnes - f_cyl_req_tonnes[-1]:.2f} T")
    ax2.text(0.05, 0.95, stats, transform=ax2.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout(); plt.show()

# --- 3. The Optimizer Engine (Triggered by Button) ---
def run_optimization(b):
    with out_status:
        clear_output()
        print("Optimizing geometry... Please wait.")
        btn_optimize.disabled = True
        
        # Parameter mapping: [Slider, Checkbox, min_bound, max_bound]
        params = {
            'L': [w_L, c_L, w_L.min, w_L.max],
            'y_f': [w_yf, c_yf, w_yf.min, w_yf.max],
            'dx_L': [w_dx_L, c_dx_L, 0.0, w_L.value], # dyn bound relies on logic below
            'dy_L': [w_dy_L, c_dy_L, w_dy_L.min, w_dy_L.max],
            'dx_U': [w_dx_U, c_dx_U, 0.0, w_L.value],
            'dy_U': [w_dy_U, c_dy_U, w_dy_U.min, w_dy_U.max],
            'theta_start': [w_theta_start, c_theta_start, w_theta_start.min, w_theta_start.max],
            'theta_end': [w_theta_end, c_theta_end, w_theta_end.min, w_theta_end.max],
            'target_dy': [w_target_dy, c_target_dy, 100.0, 2000.0]
        }
        
        param_order = list(params.keys())
        free_indices = [i for i, k in enumerate(param_order) if not params[k][1].value]
        
        if len(free_indices) == 0:
            print("All parameters held! Nothing to optimize.")
            btn_optimize.disabled = False
            return
            
        load_kg = w_load.value
        
        # Build initial guess for FREE vars
        x0 = [params[param_order[i]][0].value for i in free_indices]
        
        def get_all_vars(x_free):
            full_vars = []
            free_idx = 0
            for i, k in enumerate(param_order):
                if i in free_indices:
                    full_vars.append(x_free[free_idx])
                    free_idx += 1
                else:
                    full_vars.append(params[k][0].value)
            return full_vars
            
        def evaluate_geometry(x_free):
            L, y_f, dx_L, dy_L, dx_U, dy_U, t_start_deg, t_end_deg, target_dy = get_all_vars(x_free)
            
            t_start = np.radians(t_start_deg)
            t_end = np.radians(t_end_deg)
            theta_rad = np.linspace(t_start, t_end, 200)
            
            xc_L = L - dx_L; yc_L = dy_L; xc_U = dx_U; yc_U = dy_U
            xl = xc_L * np.cos(theta_rad) - yc_L * np.sin(theta_rad)
            yl = xc_L * np.sin(theta_rad) + yc_L * np.cos(theta_rad)
            xu = xc_U * np.cos(theta_rad) - yc_U * np.sin(theta_rad)
            yu = y_f + xc_U * np.sin(theta_rad) + yc_U * np.cos(theta_rad)
            
            l_cyl = np.sqrt((xu - xl)**2 + (yu - yl)**2)
            y_wheel = L * np.sin(theta_rad)
            
            return L, y_f, target_dy, theta_rad, l_cyl, y_wheel, xl, yl, xu, yu, dx_L, dy_L, dx_U, dy_U

        def objective(v):
            L, y_f, target_dy, theta_rad, l_cyl, y_wheel, xl, yl, xu, yu, dx_L, dy_L, dx_U, dy_U = evaluate_geometry(v)
            dy = np.diff(y_wheel)
            dl = np.diff(l_cyl); dl[dl == 0] = 1e-6
            mech_ratio = np.abs(dy / dl)
            symmetry_penalty = ((dx_L - dx_U)**2 + (dy_L + dy_U)**2) * 1e-6
            return mech_ratio[-1] + symmetry_penalty

        CYL_HALF_W = 152.4 / 2.0
        ARM_HALF_W = 100.0 / 2.0
        MIN_CLEARANCE = CYL_HALF_W + ARM_HALF_W

        best_S = None; best_res = None; best_force = float('inf')
        standard_strokes = [8, 10, 12, 16, 18, 24, 36, 48]
        
        for S in standard_strokes:
            min_l = (12.25 + S) * 25.4
            max_l = (12.25 + 2*S) * 25.4
                
            def get_min_dist(v):
                L, y_f, target_dy, theta_rad, l_cyl, y_wheel, xl, yl, xu, yu, dx_L, dy_L, dx_U, dy_U = evaluate_geometry(v)
                p_cyl1 = np.vstack([xl, yl])
                p_cyl2 = np.vstack([xu, yu])
                
                p_armL1 = np.zeros((2, len(theta_rad)))
                p_armL2 = np.vstack([L * np.cos(theta_rad), L * np.sin(theta_rad)])
                
                p_armU1 = np.zeros((2, len(theta_rad))); p_armU1[1, :] = y_f
                p_armU2 = np.vstack([L * np.cos(theta_rad), y_f + L * np.sin(theta_rad)])
                
                dist1 = segment_distance(p_cyl1, p_cyl2, p_armL1, p_armL2)
                dist2 = segment_distance(p_cyl1, p_cyl2, p_armU1, p_armU2)
                return min(np.min(dist1), np.min(dist2))

            def check_travel(v):
                L, y_f, target_dy, theta_rad, l_cyl, y_wheel, xl, yl, xu, yu, dx_L, dy_L, dx_U, dy_U = evaluate_geometry(v)
                travel = np.abs(y_wheel[-1] - y_wheel[0])
                return travel - target_dy

            constraints = [
                {'type': 'ineq', 'fun': lambda v, m_l=max_l: m_l - np.max(evaluate_geometry(v)[4])}, # Max length
                {'type': 'ineq', 'fun': lambda v, min_l=min_l: np.min(evaluate_geometry(v)[4]) - min_l}, # Min length
                {'type': 'ineq', 'fun': lambda v, s_mm=(S*25.4): s_mm - (np.max(evaluate_geometry(v)[4]) - np.min(evaluate_geometry(v)[4]))}, # Stroke
                {'type': 'ineq', 'fun': lambda v: get_min_dist(v) - MIN_CLEARANCE}, # Physical clearance
                {'type': 'ineq', 'fun': lambda v: check_travel(v) + 5.0}, # Travel >= target - 5mm
                {'type': 'ineq', 'fun': lambda v: 5.0 - check_travel(v)}  # Travel <= target + 5mm
            ]
            
            bounds = []
            for i in free_indices:
                k = param_order[i]
                if k in ['dx_L', 'dx_U']:
                    bounds.append((0, 5000)) # Upper limit handled by geometric limits softly or we just allow it loosely
                else:
                    bounds.append((params[k][2], params[k][3]))
                    
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
            
            if res.success and res.fun < best_force:
                best_force = res.fun; best_res = res; best_S = S

        if best_res:
            full_opt = get_all_vars(best_res.x)
            for i, k in enumerate(param_order):
                if i in free_indices:
                    params[k][0].value = full_opt[i]
            print(f"Success! Optimized for {best_S}in stroke cylinder.")
        else:
            print("Failed to find valid geometry within constraints.")
            
        btn_optimize.disabled = False

btn_optimize.on_click(run_optimization)

# --- 4. Layout and Display ---
out_plot = widgets.interactive_output(update_dashboard, {
    'L': w_L, 'y_f': w_yf, 'dx_L': w_dx_L, 'dy_L': w_dy_L, 'dx_U': w_dx_U, 'dy_U': w_dy_U, 
    'theta_start': w_theta_start, 'theta_end': w_theta_end, 'load_kg': w_load,
    'cyl_diam': w_cyl_diam, 'nom_press': w_nom_press, 'losses': w_losses, 'line_fric': w_line_fric
})

ui_layout = widgets.VBox([
    widgets.HBox([w_L, c_L, w_yf, c_yf]),
    widgets.HBox([w_load, w_target_dy, c_target_dy]),
    widgets.HBox([w_dx_L, c_dx_L, w_dy_L, c_dy_L]),
    widgets.HBox([w_dx_U, c_dx_U, w_dy_U, c_dy_U]),
    widgets.HBox([w_theta_start, c_theta_start, w_theta_end, c_theta_end]),
    widgets.HBox([w_cyl_diam, w_nom_press, w_losses, w_line_fric]),
    widgets.HBox([btn_optimize]),
    out_status,
    out_plot
])

display(ui_layout)