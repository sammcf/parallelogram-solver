# Parallelogram Linkage Solver: Design Notes

This document captures the design intent and decision-making process behind the solver's architecture, fitness landscape, and GA parameter choices.

## 1. Problem Domain

We are optimising the geometry of a four-bar linkage (generalised parallelogram) to achieve a target vertical travel under load, actuated by a hydraulic cylinder selected from a discrete catalogue of standard stroke sizes.

The system has ~11 coupled parameters (arm lengths, frame heights, frame offset, topology, lug positions, stroke selection) and multiple hard physical constraints (cylinder spec window, mechanical clearance) plus soft preferences (force margin, parallelism, stroke preference). The fitness landscape is therefore highly multimodal, with large infeasible regions and narrow corridors of viable geometry.

## 2. Coordinate Convention: Endpoint-Local Lugs

### The decision
Arm-mounted cylinder lugs ('L' for lower arm, 'U' for upper arm) use an **endpoint-local** coordinate system:
- **Origin** at the arm's effector-side endpoint (P3 for lower, P4 for upper)
- **u** = distance back along the arm toward the frame pivot
- **v** = perpendicular offset from the arm centreline

### Why
The lug's mechanical purpose is to push/pull near the end of the arm where it has the most leverage. Parameterising from the endpoint means `u` directly represents "how far back from the tip" — a quantity with clear physical meaning. It also makes the symmetry constraint (section 3) natural to express.

The implementation negates `u` in `get_lug_pos` because the arm's angle vector points from frame pivot toward endpoint, so walking "back along the arm" is the negative direction: `get_lug_pos(endpoint, angle, -u, v)`.

### What went wrong before
The original system used frame-pivot-origin coordinates. This made the symmetry mapping confusing and led to a bug where both lugs ended up at the same end of the parallelogram, producing a short cylinder that intersected both arms.

## 3. LU Symmetry: 180-Degree Rotation

### The decision
For the diagonal (Lower-to-Upper) topology with symmetrical lugs enabled, the second lug is derived from the first by the 180° rotational symmetry of the parallelogram:
```
u2 = (1 - frac) * L_U    (complementary position)
v2 = -v1                  (opposite perpendicular offset)
```

### Why
A true parallelogram has 180° rotational symmetry about its centre. This symmetry maps diagonal corners: P3↔P2 and P1↔P4. If lug1 is at fraction `frac` from P3, the symmetric lug2 must be at fraction `frac` from P2 — which is `(1 - frac)` from P4 in our endpoint-local convention. The perpendicular offset flips sign because the rotation reverses the normal direction.

This constraint halves the lug search space (4 free parameters → 2) and guarantees a diagonal actuator arrangement, which is the most common configuration for parallelogram lifts.

### What went wrong before
An early implementation used `u2 = frac * L_U` (same fraction from the same end), which placed both lugs near the effector endpoints. This produced a short, near-horizontal cylinder that couldn't generate useful travel and violated clearance constraints everywhere.

## 4. Fitness Landscape Design

The central challenge of this GA is that most of the parameter space produces physically impossible or useless linkages. The fitness function must simultaneously:
1. Kill obviously infeasible solutions (hard constraints)
2. Provide smooth gradient toward feasible regions (soft modifiers)
3. Not create flat plateaus where the GA has no signal to follow

### 4.1. Hard Constraints (Death Penalties)

These return `0.0001` immediately — the genome does not survive:

- **Impossible geometry**: arm too short to achieve target travel (`sin(theta_start)` outside [-1, 1])
- **Kinematic failure**: `analyze_range` throws an exception (no valid positions exist)
- **Clearance violation**: cylinder mid-span passes within `(arm_width/2 + cyl_envelope/2)` of either arm centreline

Death penalties are appropriate here because these represent physical impossibilities. There is no "almost clears" — either the cylinder fits between the arms or it doesn't. Giving gradient to near-misses would waste GA effort exploring geometries that can never be built.

### 4.2. The Rubber Wall (Cylinder Spec)

```python
over_ext = max(0, l_max - max_spec)
over_ret = max(0, min_spec - l_min)
wall_mod = exp(-0.01 * (over_ext² + over_ret²))
```

This replaced an earlier death penalty for cylinder spec violations. The exponential decay provides:
- **No penalty** when fully within spec (`wall_mod = 1.0`)
- **Gentle nudge** for small violations: 5mm over → `wall_mod ≈ 0.78`
- **Strong suppression** for large violations: 20mm → `0.018`, 30mm → `0.0001`

#### Why not a death penalty?
The cylinder spec window is a narrow target. With a death penalty, the GA had to randomly stumble into geometries where the cylinder range happened to land inside the spec — a needle-in-haystack search with no gradient. The rubber wall lets solutions that are *close* to spec survive and breed, converging the population toward the boundary from outside.

The tradeoff is that returned solutions can be slightly out of spec (up to ~30mm before fitness drops to death-penalty levels). This is acceptable because the user sees the spec status in the UI metrics and can manually adjust. The GA's job is to find the *region*, not the exact point.

#### Why quadratic exponent?
The squared term `over²` makes the penalty accelerate — gentle near the boundary, savage further out. A linear exponent (`exp(-k * over)`) would decay too uniformly, not distinguishing between "1mm over" and "15mm over" strongly enough.

### 4.3. Soft Modifiers

Each modifier is a `[0, 1]` multiplier on the base fitness, using smooth hyperbolic or exponential decay. They are multiplicative so that any single bad property can suppress fitness, but a solution that's good on all axes gets their combined benefit.

| Modifier | Formula | Purpose |
|---|---|---|
| `travel_fit` | `2000 / (1 + max(0, \|error\| - 50))` | Base score. ±50mm dead zone so near-hits aren't penalised. |
| `kin_mod` | `valid_fraction` | Fraction of the sweep with valid kinematics. Prefers robust geometry. |
| `force_mod` | `1 / (1 + excess/2)` | Penalises peak force exceeding cylinder capacity. |
| `ratio_mod` | `1 / (1 + max(0, ratio-8)*2)` | Penalises extreme mechanical ratios (>8:1). |
| `parallel_mod` | `1 / (1 + (\|ΔL\| + \|ΔH\|) / 200)` | Prefers true parallelograms. |
| `stroke_mod` | `0.3^(\|idx_distance\|)` | Optional. Exponential preference for a nominated cylinder stroke. |
| `wall_mod` | `exp(-0.01 * over²)` | Rubber wall for cylinder spec (see above). |

#### Why `1/(1 + x)` not `exp(-x)`?
The hyperbolic form has a long tail — it never fully kills a solution, just progressively discounts it. This is important because these are *preferences*, not physical constraints. A solution with a 10:1 mechanical ratio is bad but not impossible. The GA should deprioritise it, not eliminate it.

#### Why multiplicative composition?
Additive fitness (weighted sum) allows a solution to compensate for a terrible property by being excellent on another. Multiplicative composition means every modifier must be reasonable — you can't offset bad clearance with great travel. This matches the engineering reality: a linkage that achieves perfect travel but exceeds cylinder capacity is useless.

### 4.4. What failed before (cliff penalties)
The original fitness function used large additive penalties (5000x, 100000x multipliers) subtracted from a ~2000 base score. This created a landscape where virtually every solution scored `0.0001` (the floor), because even a single moderate penalty exceeded the base fitness by orders of magnitude. The GA had no gradient — every bad solution looked identical. Replacing these with smooth `[0, 1]` multipliers immediately produced useful convergence.

## 5. Dynamic Kinematic Sweep

### The decision
Instead of evaluating the linkage over a fixed angular range (-45° to 45°), the sweep is computed dynamically:
```python
theta_end = 20°  (typical transit height)
theta_start = arcsin(sin(20°) - target_travel / L_L)
```

### Why
The fixed range was arbitrary and disconnected from the actual design target. A linkage with short arms evaluated over -45° to 45° would show huge travel that was irrelevant to the 1000mm target. A linkage with long arms might show insufficient travel over the same range because the interesting region was elsewhere.

The dynamic sweep ensures the GA evaluates exactly the range that would produce the target travel for the given arm length. This makes the travel fitness modifier essentially a measure of "does this geometry *work* over this range" rather than "does this geometry *happen to produce the right number* over an arbitrary range." It dramatically improves convergence because the GA no longer wastes effort optimising for the wrong sweep.

## 6. Clearance: Pin Inset

### The decision
Before checking cylinder-to-arm clearance, the cylinder segment is inset by 10% from both ends:
```python
Q_vec = Q2 - Q1
Q1_safe = Q1 + (Q_vec * 0.1)
Q2_safe = Q2 - (Q_vec * 0.1)
```

### Why
The cylinder is physically mounted to the arms at the lug points. At those exact locations, the cylinder and arm are necessarily in contact (that's where the pin goes). Checking clearance at the mounting pins always reports zero distance, triggering the death penalty for every possible geometry.

By insetting the check segment, we test the mid-span of the cylinder where actual collisions would occur — the body of the cylinder passing through an arm it shouldn't intersect. 10% was chosen as a conservative margin that excludes the pin region without ignoring too much of the cylinder length.

## 7. GA Parameter Choices

### 7.1. Blend Crossover

```python
alpha = uniform(0.25, 0.75)
offspring = round(alpha * parent_a + (1-alpha) * parent_b)
```

#### Why not single-point crossover?
Single-point crossover splits the genome at a random position and swaps tails. For our genome `[L_L, L_U, H_f, H_e, dx_f, topo, l1u, l1v, l2u, l2v, stroke]`, the parameters are tightly coupled — lug positions are only meaningful relative to arm lengths, and stroke selection depends on the overall geometry. Splitting at position 4 takes geometry from one parent and lug positions from another, producing offspring that are almost certainly worse than either parent.

#### Why not uniform crossover?
Uniform crossover independently selects each gene from either parent. While better than single-point (no arbitrary split), it still produces combinations where `L_L` comes from parent A but `l1u` (which is a fraction of `L_L`) comes from parent B. The resulting lug position is nonsensical for the inherited arm length.

#### Why blend/arithmetic crossover?
Averaging (with random weight) preserves the correlated structure of the genome. If parent A has `(L_L=800, l1u=200)` (25% lug position) and parent B has `(L_L=1200, l1u=300)` (25% lug position), their blend produces `(L_L≈1000, l1u≈250)` — still a 25% lug position. The proportional relationships survive crossover.

The random alpha in [0.25, 0.75] rather than fixed 0.5 maintains population diversity. A fixed average would collapse the population to a single centroid; the variable weight lets offspring spread across the region between parents.

### 7.2. Adaptive Mutation [15, 5]

- **Low-fitness solutions**: mutate 15% of genes (~2 of 11). These solutions are far from any optimum and need significant perturbation to explore.
- **High-fitness solutions**: mutate 5% of genes (~1 of 11). These are near an optimum and need gentle refinement, not demolition.

#### Why not fixed 20%?
With 11 genes, 20% mutation means ~2 genes are randomly altered every generation. For a solution that's converging toward an optimum (`L_L=1043, H_f=475`), randomly perturbing 2 genes by replacement would frequently produce offspring that are dramatically worse. This is anti-productive — we want the GA to *refine*, not *thrash*.

#### Why `mutation_by_replacement=False`?
With replacement enabled (the default), a mutated gene gets a completely new random value drawn uniformly from its gene space. For `L_L` with range [400, 1500], a solution at `L_L=1043` could jump to `L_L=400` in a single mutation — a 60% change that almost certainly destroys fitness.

With replacement disabled, mutation is *additive* — a small random value is added to the existing gene. This produces local perturbation: `L_L=1043` might become `L_L=1060` or `L_L=1025`. This is the correct behaviour for continuous optimisation of physical parameters.

### 7.3. Elitism and Selection

- **`keep_elitism=10`**: The top 10 solutions are preserved across generations, ensuring the best-known geometry is never lost to crossover or mutation.
- **`keep_parents=0`**: Parents are not preserved with stale fitness values. Without this, PyGAD can carry forward solutions evaluated under different conditions (e.g., before a constraint was added) with outdated fitness scores.
- **`parent_selection_type="sss"`** (steady-state): Combined with elitism, this provides strong selection pressure while the blend crossover and adaptive mutation handle exploration.
- **Post-hoc re-evaluation**: Before returning results, every candidate is freshly evaluated. This catches any solutions that survived with stale fitness from elitism or other PyGAD internals.

### 7.4. Population and Mating

- **`sol_per_pop=200`**: Large enough to cover the search space initially. With 11 genes and wide ranges, 200 individuals provides reasonable coverage without excessive evaluation cost.
- **`num_parents_mating=25`**: ~12% of the population breeds each generation. This is moderate — enough parents to maintain diversity, few enough to maintain selection pressure.

## 8. Cylinder Catalogue

Standard hydraulic cylinder strokes: `[8, 10, 12, 16, 18, 24, 36, 48]` inches.

For a given stroke `S`:
- **Minimum length** (fully retracted): `(12.25 + S) * 25.4 mm`
- **Maximum length** (fully extended): `min_length + S * 25.4 mm`

The 12.25-inch base accounts for the piston rod, seals, and end caps. This is an industry-standard approximation for tie-rod cylinders.

### Stroke Preference

When enabled, an exponential decay `0.3^|distance|` penalises solutions that select strokes far from the user's preferred size. One step away (e.g., preferring 12" but getting 16") reduces fitness to 30%. Two steps away reduces to 9%. This strongly converges the GA toward the preferred stroke while still allowing it to explore adjacent sizes if the geometry demands it.

The preference is optional — disabling it lets the GA find the globally optimal stroke size regardless of user preference.

## 9. Future Work

### 9.1. Extract Shared Domain Layer

Genome layout, lug parameterisation, and decode logic are duplicated between `optimizer.py` (GA encode/decode) and `app.py` (widget ↔ genome mapping in `_apply_solution` + lug config sections). Adding or reordering a gene requires updating hard-coded indices in both files — a fragile arrangement that has already caused bugs.

Extract into a shared module (e.g. `genome.py` or `params.py`) that defines: gene indices, gene space ranges, encode/decode functions for lug params, and topology-to-member mappings. Both frontend and solver import from this single source of truth.

### 9.2. Effector Path Conditioning

For a perfect parallelogram (`L_L == L_U`, `H_f == H_e`), the effector endpoint traces a pure vertical line. Modified linkages — where arm lengths or frame dimensions differ — produce a non-linear arc instead. Currently the solver is agnostic to the shape of this arc; it only cares about total vertical travel.

Adding path-shape parameters would let users condition the GA on the effector's trajectory for a given geometry package:
- **Max lateral deviation**: constrain how far the effector wanders horizontally from a vertical line (e.g. "must stay within ±15mm of vertical over the full stroke").
- **Arc radius bounds**: set minimum/maximum curvature of the path, useful when the effector must track along or clear a particular surface.
- **Target path profile**: supply an explicit path (series of x,y points or a parametric curve) and penalise deviation from it, for cases where the effector must follow a specific trajectory (e.g. matching an existing guide rail or mating geometry).

This would be implemented as additional soft fitness modifiers — evaluate the effector path at each step of the kinematic sweep, compute the relevant deviation metric, and apply a `1/(1+x)` style penalty. The user-facing controls would be optional toggles with threshold inputs, similar to the existing clearance and stroke preference controls.

### 9.3. Performance Optimisation Pass

The current implementation prioritises correctness and readability. Stay in Python; target hot-path improvements:
- **Vectorise `analyze_range`**: the inner kinematic loop is pure Python. Rewriting as vectorised NumPy (compute all angles/positions as arrays in one pass) would be the single biggest win — this is called ~60k times per optimisation run.
- **Batch fitness evaluation**: evaluate the entire population's fitness in one vectorised call rather than looping per-genome. NumPy broadcasting over the population axis where possible.
- **Integer-only genome**: all gene values can be integers (mm precision is sufficient for this domain). This simplifies memoisation — genome tuples are directly hashable with no float-rounding ambiguity — and eliminates floating-point reproducibility issues in crossover/mutation.
- **Memoisation**: elites are re-evaluated each generation. With integer genomes, cache fitness by genome tuple directly. `functools.lru_cache` or a simple dict.
- **Numba JIT**: if vectorisation alone isn't enough, `@numba.njit` on the inner kinematics loop avoids the Cython build step while getting near-C speed. NumPy-style code JITs cleanly.
- **Streamlit rendering**: the matplotlib plot regenerates from scratch on every parameter change. Investigate partial updates or a lighter backend (e.g. Plotly with WebGL) for interactive tweaking.

A Haskell rewrite was considered and is architecturally viable (the fitness pipeline and genome ADTs are a natural fit), but the ecosystem overhead isn't justified while the Python version is functional. Revisit if performance becomes a blocking constraint or if the project scope grows significantly.
