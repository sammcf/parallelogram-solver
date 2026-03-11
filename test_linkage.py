import numpy as np
from linkage import FourBarLinkage

def test_true_parallelogram():
    # Parameters from notebook defaults
    L = 795.0
    yf = 350.0
    dx_L = 25.0
    dy_L = 150.0
    dx_U = 25.0
    dy_U = -150.0
    theta_start = -30.0
    theta_end = 20.0

    # Lug coords are in endpoint-local frame:
    #   origin at arm endpoint (P3 for lower, P4 for upper),
    #   u = distance back along arm toward frame pivot,
    #   v = perpendicular offset (same sign convention as before).
    #
    # Old notebook used frame-pivot origin, so the conversion is:
    #   u_new = L - u_old,  v_new = v_old

    linkage = FourBarLinkage(L, L, yf, yf)

    lug1 = ('L', dx_L, dy_L)          # 25mm from P3, 150mm perp
    lug2 = ('U', L - dx_U, dy_U)      # 770mm from P4, -150mm perp

    results = linkage.analyze_range(theta_start, theta_end, lug1, lug2)
    
    print(f"Min Cylinder Length: {np.min(results['l_cyl']):.2f} mm")
    print(f"Max Cylinder Length: {np.max(results['l_cyl']):.2f} mm")
    print(f"Stroke: {np.max(results['l_cyl']) - np.min(results['l_cyl']):.2f} mm")
    print(f"Travel: {results['y_effector'][-1] - results['y_effector'][0]:.2f} mm")
    print(f"Max Mech Ratio: {np.max(results['mech_ratio']):.2f}")

if __name__ == "__main__":
    test_true_parallelogram()
