# Parallelogram Linkage Solver: Genetic Algorithm Evolution

This document records the design decisions and thinking process for evolving the parallelogram linkage solver into a generalized, genetic-algorithm-driven optimization tool.

## 1. Problem Generalization

### 1.1. Geometry: Beyond the True Parallelogram
The current solver assumes a true parallelogram where:
- Upper and lower arm lengths are identical ($L$).
- The vertical distance between frame pivots ($y_f$) equals the distance between end-effector pivots.

To generalize, we will move towards a **Four-Bar Linkage** model:
- **$L_{lower}$**: Length of the lower arm.
- **$L_{upper}$**: Length of the upper arm.
- **$H_{frame}$**: Vertical distance between fixed pivots on the frame.
- **$H_{effector}$**: Vertical distance between pivots on the moving end-link.
- **$\Delta X_{frame}$**: Horizontal offset between frame pivots (allowing non-vertical frame mounting).

### 1.2. Mathematical Model: Four-Bar Position Analysis
To support modified parallelograms, we use the standard four-bar linkage equations. Given the lower arm angle $\theta$:
1.  **Lower End Pivot ($P_3$):** $(L_L \cos \theta, L_L \sin \theta)$.
2.  **Upper End Pivot ($P_4$):** Must satisfy $|P_4 - P_3| = H_e$, where $P_4 = (x_f + L_U \cos \phi, y_f + L_U \sin \phi)$.
3.  **End-Link Angle ($\psi$):** $\operatorname{atan2}(y_4 - y_3, x_4 - x_3)$.

### 1.3. Lug Coordinate Systems
Lugs are defined in local "member-space" $(u, v)$ where $u$ is along the member's longitudinal axis and $v$ is perpendicular:
-   **Arm Lugs:** Origin at frame pivot, $u$-axis pointing towards the end-effector pivot.
-   **Frame Lugs:** Fixed global coordinates $(x, y)$.
-   **End-Link Lugs:** Origin at $P_3$, $u$-axis pointing towards $P_4$.

### 1.4. Actuator Mounting Arrangements
The solver should support multiple "topologies":
1.  **Diagonal (Current):** Lower Arm to Upper Arm.
2.  **Frame to Arm:** Frame (fixed point) to either Lower or Upper Arm.
3.  **Crossed:** Lower Arm to Frame, or Frame to Upper Arm in a way that crosses the other arm.
4.  **Indirect:** Using a bell crank (likely out of scope for phase 1).

## 2. Genetic Algorithm Approach

### 2.1. The Genome (Solution Space)
A individual solution will be represented by a vector of parameters:
- **Geometry:** $L_L, L_U, H_f, H_e, \Delta X_f$.
- **Mounting:** $TopologyID, Lug1_{member}, Lug1_{u}, Lug1_{v}, Lug2_{member}, Lug2_{u}, Lug2_{v}$.
- **Cylinder:** $StrokeID$ (from a catalog of standard cylinders).

### 2.2. Fitness Functions & Weights
We want to optimize for:
1.  **Capacity Margin (High Weight):** Maximize $F_{capacity} - F_{required}$ across the whole stroke.
2.  **Stroke Utilization (Medium Weight):** Minimize "wasted" cylinder stroke (percentage of $S$ used).
3.  **Mechanical Ratio Linearity (Low Weight):** Prefer linkages where the force requirement is relatively flat.
4.  **Compactness (Low Weight):** Penalize excessively long arms or lugs.
5.  **Clearance (Hard Constraint/Penalty):** Heavy penalty for any intersection between cylinder and arms.
6.  **Parallelism (User Preference):** For applications requiring the load to stay level, penalize $|H_{frame} - H_{effector}|$ and $|L_{upper} - L_{lower}|$.

## 4. Using the Tool

### 4.1. Installation
Ensure you have the required dependencies:
```bash
pip install pygad streamlit numpy matplotlib scipy
```

### 4.2. Running the Optimizer
Start the Streamlit dashboard:
```bash
streamlit run app.py
```

### 4.3. Workflow
1.  **Define Targets:** Enter the desired travel (mm) and the load (kg) in the sidebar.
2.  **Evolve:** Set the number of generations and click "Run Optimizer".
3.  **Compare:** Review the top 5 distinct solutions found by the GA. Click on the solution buttons to view their geometry and force profiles.
4.  **Refine:** Manually adjust any parameters (lengths, lug positions) in the "Details" section to see how it affects the mechanical performance in real-time.
