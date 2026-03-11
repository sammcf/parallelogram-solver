import numpy as np

def segment_distance(p1, p2, p3, p4):
    """
    Computes the minimum distance between two segments (p1-p2) and (p3-p4).
    Expects p1, p2, p3, p4 to be arrays of shape (2, N).
    """
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = np.sum(u*u, axis=0)
    b = np.sum(u*v, axis=0)
    c = np.sum(v*v, axis=0)
    d = np.sum(u*w, axis=0)
    e = np.sum(v*w, axis=0)
    D = a*c - b*b
    sN = np.zeros_like(D)
    sD = D.copy()
    tN = np.zeros_like(D)
    tD = D.copy()
    
    small = D < 1e-8
    sN[small] = 0.0
    sD[small] = 1.0
    tN[small] = e[small]
    tD[small] = c[small]
    ok = ~small
    sN[ok] = b[ok]*e[ok] - c[ok]*d[ok]
    tN[ok] = a[ok]*e[ok] - b[ok]*d[ok]
    
    s_lt_0 = ok & (sN < 0.0)
    sN[s_lt_0] = 0.0
    tN[s_lt_0] = e[s_lt_0]
    tD[s_lt_0] = c[s_lt_0]
    s_gt_1 = ok & (sN > sD)
    sN[s_gt_1] = sD[s_gt_1]
    tN[s_gt_1] = e[s_gt_1] + b[s_gt_1]
    tD[s_gt_1] = c[s_gt_1]
    
    t_lt_0 = tN < 0.0
    tN[t_lt_0] = 0.0
    sN[t_lt_0] = np.clip(-d[t_lt_0], 0.0, a[t_lt_0])
    sD[t_lt_0] = a[t_lt_0]
    t_gt_1 = tN > tD
    tN[t_gt_1] = tD[t_gt_1]
    sN[t_gt_1] = np.clip((-d[t_gt_1] + b[t_gt_1]), 0.0, a[t_gt_1])
    sD[t_gt_1] = a[t_gt_1]
    
    sc = np.where(np.abs(sD) < 1e-8, 0.0, sN / sD)
    tc = np.where(np.abs(tD) < 1e-8, 0.0, tN / tD)
    
    dP = w + (sc * u) - (tc * v)
    return np.sqrt(np.sum(dP*dP, axis=0))

class FourBarLinkage:
    def __init__(self, L_L, L_U, H_f, H_e, dx_f=0.0, dy_f=0.0):
        self.L_L = L_L
        self.L_U = L_U
        self.H_f = H_f
        self.H_e = H_e
        self.dx_f = dx_f
        self.dy_f = dy_f

        self.P1 = np.array([0.0, 0.0]) # Lower Frame Pivot
        self.P2 = np.array([dx_f, H_f + dy_f]) # Upper Frame Pivot

    def solve_positions(self, theta_rad):
        """
        Solves for P3, P4, and phi given lower arm angle theta.
        Returns P3, P4, phis, and a validity mask.
        """
        # P3: Lower Arm End
        P3 = np.array([self.L_L * np.cos(theta_rad), self.L_L * np.sin(theta_rad)])
        
        # P4: Upper Arm End
        d_vec = self.P2[:, None] - P3
        d = np.sqrt(np.sum(d_vec**2, axis=0))
        
        # Triangle sides: a=L_U, b=H_e, c=d
        denom = (2 * self.L_U * d)
        denom[denom < 1e-9] = 1e-9
        cos_alpha = (self.L_U**2 + d**2 - self.H_e**2) / denom
        
        # Validity mask: can we actually form this triangle?
        valid = (cos_alpha >= -1.0) & (cos_alpha <= 1.0)
        
        # Clamp for robustness
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        
        # Angle of vector P3-P2
        base_angle = np.arctan2(P3[1] - self.P2[1], P3[0] - self.P2[0])
        phi = base_angle + alpha
        
        P4 = np.array([self.P2[0] + self.L_U * np.cos(phi), 
                       self.P2[1] + self.L_U * np.sin(phi)])
        
        return P3, P4, phi, valid

    def get_lug_pos(self, P_pivot, angle, u, v):
        """
        Calculates global lug position from local (u, v).
        u is along the arm, v is perpendicular.
        """
        # Rotation matrix R = [[cos, -sin], [sin, cos]]
        x = P_pivot[0] + u * np.cos(angle) - v * np.sin(angle)
        y = P_pivot[1] + u * np.sin(angle) + v * np.cos(angle)
        return np.array([x, y])

    def analyze_range(self, theta_start_deg, theta_end_deg, lug1_config, lug2_config, n_steps=500):
        """
        lug1_config: (member, u, v)
        For 'L'/'U' members, u is distance from arm endpoint (P3/P4) back
        along the arm toward the frame pivot, v is perpendicular offset.
        """
        thetas = np.radians(np.linspace(theta_start_deg, theta_end_deg, n_steps))
        P3, P4, phis, valid = self.solve_positions(thetas)

        if not np.any(valid):
            raise ValueError("No valid kinematic positions in range")

        def resolve_lug(config, t, p3, p4, ph):
            member, u, v = config
            if member == 'L':
                # Origin at arm endpoint P3, u back along arm toward P1
                return self.get_lug_pos(p3, t, -u, v)
            elif member == 'U':
                # Origin at arm endpoint P4, u back along arm toward P2
                return self.get_lug_pos(p4, ph, -u, v)
            elif member == 'F':
                return np.array([np.full_like(t, u), np.full_like(t, v)])
            elif member == 'E':
                dx = p4[0] - p3[0]
                dy = p4[1] - p3[1]
                psi = np.arctan2(dy, dx)
                return self.get_lug_pos(p3, psi, u, v)
            return None

        # Slice results by validity
        t_v = thetas[valid]
        p3_v = P3[:, valid]
        p4_v = P4[:, valid]
        ph_v = phis[valid]

        Q1 = resolve_lug(lug1_config, t_v, p3_v, p4_v, ph_v)
        Q2 = resolve_lug(lug2_config, t_v, p3_v, p4_v, ph_v)
        
        l_cyl = np.sqrt(np.sum((Q2 - Q1)**2, axis=0))
        y_effector = p3_v[1]
        
        dy_eff = np.gradient(y_effector)
        dl_cyl = np.gradient(l_cyl)
        
        dl_cyl[np.abs(dl_cyl) < 1e-3] = 1e-3 # Minimum movement threshold
        mech_ratio = np.abs(dy_eff / dl_cyl)
        
        return {
            'thetas': t_v,
            'phis': ph_v,
            'P3': p3_v,
            'P4': p4_v,
            'Q1': Q1,
            'Q2': Q2,
            'l_cyl': l_cyl,
            'y_effector': y_effector,
            'mech_ratio': mech_ratio,
            'valid_fraction': np.mean(valid)
        }
