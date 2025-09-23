# RiteWeight: Randomized Iterative Trajectory Reweighting

**RiteWeight** is a lightweight Python package for computing steady-state weights from short molecular dynamics (MD) trajectory segments.

It uses a simple yet powerful approach: at each iteration, the method randomly clusters a reduced feature space (e.g., PCA, tICA, VAMP) 
and reweights the trajectory based on the stationary distribution derived from the transition matrix. 
This randomized clustering approach allows RiteWeight to efficiently estimate long-timescale kinetics and thermodynamics from short simulations 
— without requiring long trajectories or predefined discrete states.

---
