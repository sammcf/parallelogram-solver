import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy.optimize import minimize

# --- 1. Define UI Widgets ---
style = {'description_width': 'initial'}
layout = widgets.Layout(width='350px')

# Manual Override Sliders
w_L = widgets.FloatSlider(value=1600.0, min=1000.0, max=2500.0, step=10.0, description='Arm Length (L)', style=style, layout=layout)
w_yf = widgets.FloatSlider(value=400.0, min=200.0, max=800.0, step=10.0, description='Frame y_f', style=style, layout=layout)
w_load = widgets.IntSlider(value=10000, min=2000, max=20000, step=1000, description='Load (kg)', style=style, layout=layout)

w_dx_L = widgets.FloatSlider(value=450.0, min=0.0, max=1500.0, step=1.0, description='Lower Lug ∥ arm (dx_L)', style=style, layout=layout)
w_dy_L = widgets.FloatSlider(value=80.0, min=-300.0, max=300.0, step=1.0, description='Lower Lug ⊥ arm (dy_L)', style=style, layout=layout)
w_dx_U = widgets.FloatSlider(value=1150.0, min=0.0, max=2000.0, step=1.0, description='Upper Lug ∥ arm (dx_U)', style=style, layout=layout)
w_dy_U = widgets.FloatSlider(value=-90.0, min=-300.0, max=300.0, step=1.0, description='Upper Lug ⊥ arm (dy_U)', style=style, layout=layout)

w_theta_start = widgets.FloatSlider(value=-30.0, min=-60.0, max=0.0, step=0.5, description='Start Angle (deg)', style=style, layout=layout)
w_theta_end = widgets.FloatSlider(value=20.0, min=0.0, max=60.0, step=0.5, description='End Angle (deg)', style=style, layout=layout)

# Optimizer Controls
w_target_dy = widgets.FloatText(value=1000.0, description='Target dY (mm):', style=style, layout=widgets.Layout(width='200px'))
btn_optimize = widgets.Button(description="Auto-Optimize Lugs", button_style='success', icon='cogs')
out_status = widgets.Output()

# --- 2. The Plotting Engine (Triggered by Sliders) ---
def update_dashboard(L, y_f, dx_L, dy_L, dx_U, dy_U, theta_start, theta_end, load_kg):
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

    # Render Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wireframe
    ax1.set_title("Linkage Wireframe")
    ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)")
    ax1.grid(True, linestyle='--', alpha=0.5); ax1.axis('equal')
    
    def draw_state(idx, color, alpha_val):
        ax1.plot([0, 0], [0, y_f], 'ks-', markersize=8) 
        ax1.plot([0, L*np.cos(theta_rad[idx])], [0, L*np.sin(theta_rad[idx])], color=color, lw=4, alpha=alpha_val)
        ax1.plot([0, L*np.cos(theta_rad[idx])], [y_f, y_f + L*np.sin(theta_rad[idx])], color=color, lw=4, alpha=alpha_val)
        ax1.plot([L*np.cos(theta_rad[idx]), L*np.cos(theta_rad[idx])], 
                 [L*np.sin(theta_rad[idx]), y_f + L*np.sin(theta_rad[idx])], 'k-', lw=5, alpha=alpha_val)
        ax1.plot([x_lower[idx], x_upper[idx]], [y_lower[idx], y_upper[idx]], 'r-o', lw=3, alpha=alpha_val)

    draw_state(0, 'gray', 0.3)
    draw_state(-1, 'blue', 1.0)
    
    # Force Curve
    ax2.set_title("Required Cylinder Force vs. Travel")
    ax2.set_xlabel("Travel (mm)"); ax2.set_ylabel("Force (Tonnes)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.plot(travel_mm, f_cyl_req_tonnes, 'b-', lw=3)
    ax2.plot(travel_mm[-1], f_cyl_req_tonnes[-1], 'ro', markersize=8)
    
    # Live Readouts
    cyl_min_in = np.min(l_cyl)/25.4
    cyl_max_in = np.max(l_cyl)/25.4
    stats = (f"Travel: {travel_mm[-1]:.1f} mm\n"
             f"Cyl Stroke Req: {cyl_max_in - cyl_min_in:.2f} in\n"
             f"Min Pin-to-Pin: {cyl_min_in:.1f} in\n"
             f"Peak Force (Lifted): {f_cyl_req_tonnes[-1]:.2f} T")
    ax2.text(0.05, 0.95, stats, transform=ax2.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout(); plt.show()

# --- 3. The Optimizer Engine (Triggered by Button) ---
def run_optimization(b):
    with out_status:
        clear_output()
        print("Optimizing geometry... Please wait.")
        btn_optimize.disabled = True
        
        L = w_L.value; y_f = w_yf.value; load_kg = w_load.value
        dy_req = w_target_dy.value; theta_end_rad = np.radians(w_theta_end.value)
        
        # Calculate start angle to hit exact target travel
        sin_start = np.sin(theta_end_rad) - (dy_req / L)
        if sin_start < -1.0 or sin_start > 1.0:
            print("ERROR: Target travel impossible with current Arm Length.")
            btn_optimize.disabled = False
            return
            
        theta_start_rad = np.arcsin(sin_start)
        theta_rad = np.linspace(theta_start_rad, theta_end_rad, 200)
        
        def calc_l_cyl(vars):
            dx_L, dy_L, dx_U, dy_U = vars
            xc_L = L - dx_L
            yc_L = dy_L
            xc_U = dx_U
            yc_U = dy_U
            x_lower = (xc_L * np.cos(theta_rad)) - (yc_L * np.sin(theta_rad))
            y_lower = (xc_L * np.sin(theta_rad)) + (yc_L * np.cos(theta_rad))
            x_upper = (xc_U * np.cos(theta_rad)) - (yc_U * np.sin(theta_rad))
            y_upper = y_f + (xc_U * np.sin(theta_rad)) + (yc_U * np.cos(theta_rad))
            return np.sqrt((x_upper - x_lower)**2 + (y_upper - y_lower)**2)

        def objective(vars):
            dx_L, dy_L, dx_U, dy_U = vars
            l_cyl = calc_l_cyl(vars)
            y_wheel = L * np.sin(theta_rad)
            dy = np.diff(y_wheel)
            dl = np.diff(l_cyl); dl[dl == 0] = 1e-6
            mech_ratio = np.abs(dy / dl)
            symmetry_penalty = ((dx_L - dx_U)**2 + (dy_L + dy_U)**2) * 1e-6
            return mech_ratio[-1] + symmetry_penalty # Minimize force at the end (lifted)

        best_S = None; best_res = None; best_force = float('inf')
        standard_strokes = [8, 10, 12, 16, 18, 24, 36, 48]
        
        for S in standard_strokes:
            min_l = (12.25 + S) * 25.4
            max_l = (12.25 + 2*S) * 25.4
            
            constraints = [
                {'type': 'ineq', 'fun': lambda v, m_l=max_l: m_l - np.max(calc_l_cyl(v))},
                {'type': 'ineq', 'fun': lambda v, min_l=min_l: np.min(calc_l_cyl(v)) - min_l},
                {'type': 'ineq', 'fun': lambda v, s_mm=(S*25.4): s_mm - (np.max(calc_l_cyl(v)) - np.min(calc_l_cyl(v)))}
            ]
            
            bounds = ((0, L), (-250, 250), (0, L), (-250, 250))
            res = minimize(objective, [L/2, 50, L/2, -50], method='SLSQP', bounds=bounds, constraints=constraints)
            
            if res.success and res.fun < best_force:
                best_force = res.fun; best_res = res; best_S = S

        if best_res:
            # Snap sliders to optimal values
            w_theta_start.value = np.degrees(theta_start_rad)
            w_dx_L.value = best_res.x[0]
            w_dy_L.value = best_res.x[1]
            w_dx_U.value = best_res.x[2]
            w_dy_U.value = best_res.x[3]
            print(f"Success! Optimized for {best_S}in stroke cylinder.")
        else:
            print("Failed to find valid geometry within cylinder hard-stop constraints.")
            
        btn_optimize.disabled = False

btn_optimize.on_click(run_optimization)

# --- 4. Layout and Display ---
out_plot = widgets.interactive_output(update_dashboard, {
    'L': w_L, 'y_f': w_yf, 'dx_L': w_dx_L, 'dy_L': w_dy_L, 'dx_U': w_dx_U, 'dy_U': w_dy_U, 
    'theta_start': w_theta_start, 'theta_end': w_theta_end, 'load_kg': w_load
})

ui_layout = widgets.VBox([
    widgets.HBox([w_L, w_yf, w_load]),
    widgets.HBox([w_target_dy, w_theta_end, btn_optimize]),
    widgets.HBox([w_dx_L, w_dy_L]),
    widgets.HBox([w_dx_U, w_dy_U]),
    w_theta_start,
    out_status,
    out_plot
])

display(ui_layout)