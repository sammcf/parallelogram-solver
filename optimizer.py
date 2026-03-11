import numpy as np
import pygad
from linkage import FourBarLinkage, segment_distance

class CylinderCatalogue:
    STROKES_IN = [8, 10, 12, 16, 18, 24, 36, 48]
    @staticmethod
    def get_specs(idx):
        S = CylinderCatalogue.STROKES_IN[idx]
        min_l = (12.25 + S) * 25.4
        max_l = min_l + (S * 25.4)
        return min_l, max_l, S

class FitnessEvaluator:
    STROKE_DECAY = 0.3  # per catalogue step from preferred

    def __init__(self, target_travel, load_kg, cyl_params, symmetrical_lugs=False, arm_width=100.0, cyl_env=152.4, use_clearance=True, preferred_stroke_idx=None):
        self.target_travel = target_travel
        self.load_kg = load_kg
        self.cyl_diam_in = cyl_params['cyl_diam_in']
        self.nom_press_psi = cyl_params['nom_press_psi']
        self.losses = cyl_params.get('losses', 0.01)
        self.line_fric = cyl_params.get('line_fric', 0.15)
        self.symmetrical_lugs = symmetrical_lugs
        self.arm_width = arm_width
        self.cyl_env = cyl_env
        self.use_clearance = use_clearance
        self.preferred_stroke_idx = preferred_stroke_idx
        
        final_pressure_psi = self.nom_press_psi * (1.0 - self.losses) * (1.0 - self.line_fric)
        piston_area_in2 = np.pi * (self.cyl_diam_in / 2.0)**2
        cyl_force_lbf = final_pressure_psi * piston_area_in2
        self.cyl_capacity_tonnes = (cyl_force_lbf * 4.44822) / (1000 * 9.81)

    def decode_genome(self, genome):
        """
        Lug u/v are in endpoint-local coords: u = distance from arm endpoint
        (P3/P4) back along the arm, v = perpendicular offset.
        """
        (L_L, L_U, H_f, H_e, dx_f, topo, l1u_p, l1v_p, l2u_p, l2v_p, s_idx) = genome
        frac1 = l1u_p / 1000.0
        u1 = frac1 * L_L
        v1 = ((l1v_p / 1000.0) * 600.0) - 300.0
        if int(topo) == 0:
            if self.symmetrical_lugs:
                u2, v2 = (1 - frac1) * L_U, -v1  # opposite end for diagonal
            else:
                u2, v2 = (l2u_p / 1000.0) * L_U, ((l2v_p / 1000.0) * 600.0) - 300.0
            l1, l2 = ('L', u1, v1), ('U', u2, v2)
        elif int(topo) == 1:
            l1, l2 = ('F', ((l1u_p/1000.0)*1500)-750, ((l1v_p/1000.0)*1500)-750), ('L', (l2u_p/1000.0)*L_L, ((l2v_p/1000.0)*600)-300)
        else:
            l1, l2 = ('F', ((l1u_p/1000.0)*1500)-750, ((l1v_p/1000.0)*1500)-750), ('U', (l2u_p/1000.0)*L_U, ((l2v_p/1000.0)*600)-300)
        return L_L, L_U, H_f, H_e, dx_f, l1, l2, int(s_idx)

    def evaluate(self, genome):
        L_L, L_U, H_f, H_e, dx_f, lug1, lug2, s_idx = self.decode_genome(genome)
        linkage = FourBarLinkage(L_L, L_U, H_f, H_e, dx_f)
        min_spec, max_spec, _ = CylinderCatalogue.get_specs(s_idx)
        try: results = linkage.analyze_range(-45, 45, lug1, lug2, n_steps=100)
        except Exception: return 0.0001

        # Hard constraint: clearance death penalty
        if self.use_clearance:
            N = results['Q1'].shape[1]
            p_frame_lower = np.zeros((2, N))
            p_frame_upper = np.tile(linkage.P2[:, None], (1, N))
            dist_L = segment_distance(results['Q1'], results['Q2'], p_frame_lower, results['P3'])
            dist_U = segment_distance(results['Q1'], results['Q2'], p_frame_upper, results['P4'])
            req_d = (self.arm_width / 2.0) + (self.cyl_env / 2.0)
            if min(np.min(dist_L), np.min(dist_U)) < req_d:
                return 0.0001

        # Hard constraint: cylinder must stay within physical spec
        l_min, l_max = np.min(results['l_cyl']), np.max(results['l_cyl'])
        if l_min < min_spec or l_max > max_spec:
            return 0.0001

        # Base fitness: closeness to target travel (±50mm dead zone)
        actual_travel = results['y_effector'][-1] - results['y_effector'][0]
        travel_err = max(0, abs(actual_travel - self.target_travel) - 50.0)
        travel_fit = 2000.0 / (1.0 + travel_err)

        # Soft modifiers: smooth 0-to-1 gradients the GA can follow
        kin_mod = results['valid_fraction']

        peak_f = np.max(((self.load_kg * 9.81) * results['mech_ratio']) / (1000 * 9.81))
        force_excess = max(0, peak_f - self.cyl_capacity_tonnes)
        force_mod = 1.0 / (1.0 + force_excess / 2.0)

        max_ratio = np.max(results['mech_ratio'])
        ratio_mod = 1.0 / (1.0 + max(0, max_ratio - 8.0) * 2.0)

        parallel_mod = 1.0 / (1.0 + (abs(L_L - L_U) + abs(H_f - H_e)) / 200.0)

        stroke_mod = 1.0
        if self.preferred_stroke_idx is not None:
            stroke_mod = self.STROKE_DECAY ** abs(s_idx - self.preferred_stroke_idx)

        return max(0.0001, travel_fit * kin_mod * force_mod * ratio_mod * parallel_mod * stroke_mod)

class GAOptimizer:
    def __init__(self, target_travel, load_kg, cyl_params, fixed_params=None, allowed_topologies=None, symmetrical_lugs=False, arm_width=100.0, cyl_env=152.4, use_clearance=True, preferred_stroke_idx=None):
        self.evaluator = FitnessEvaluator(target_travel, load_kg, cyl_params, symmetrical_lugs, arm_width, cyl_env, use_clearance, preferred_stroke_idx)
        self.fixed_params, self.allowed_topos = fixed_params or {}, allowed_topologies or [0, 1, 2]
        
    def fitness_func(self, ga, sol, idx): return self.evaluator.evaluate(sol)

    def run(self, gens=100):
        gene_space = [{'low': 400, 'high': 1500}, {'low': 400, 'high': 1500}, {'low': 200, 'high': 600}, {'low': 200, 'high': 600}, 
                      {'low': -200, 'high': 200}, self.allowed_topos, {'low': 0, 'high': 1000}, {'low': 0, 'high': 1000}, 
                      {'low': 0, 'high': 1000}, {'low': 0, 'high': 1000}, list(range(len(CylinderCatalogue.STROKES_IN)))]
        param_map = ['L_L', 'L_U', 'H_f', 'H_e', 'dx_f', 'topo', 'l1u_pct', 'l1v_pct', 'l2u_pct', 'l2v_pct', 'stroke_idx']
        for i, name in enumerate(param_map):
            if name in self.fixed_params: gene_space[i] = [int(self.fixed_params[name])]
        
        ga = pygad.GA(num_generations=gens, num_parents_mating=25, fitness_func=self.fitness_func, sol_per_pop=200, num_genes=len(gene_space), gene_space=gene_space, parent_selection_type="sss", keep_parents=0, keep_elitism=10, crossover_type="single_point", mutation_type="random", mutation_percent_genes=20, gene_type=int)
        ga.run()
        
        sorted_idx = np.argsort(ga.last_generation_fitness)[::-1]
        unique, seen = [], []
        for idx in sorted_idx:
            sol = ga.population[idx]
            # Re-evaluate to catch any stale fitness values
            fresh_fit = self.evaluator.evaluate(sol)
            if fresh_fit <= 0.0001:
                continue
            if not any(np.linalg.norm(sol - s) < 150 for s in seen):
                unique.append((sol, fresh_fit))
                seen.append(sol)
            if len(unique) >= 5: break
        return unique
