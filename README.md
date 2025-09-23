# rite_weight: Randomized Iterative Trajectory Reweighting
Authors: Sagar Kania, Robert J. Webber, Gideon Simpson, David Aristoff, and Daniel M. Zuckerman

# Background
**rite_weight** is a lightweight Python package for computing steady-state weights from short molecular dynamics (MD) trajectory segments.

It uses a simple yet powerful approach: at each iteration, the method randomly clusters a reduced feature space (e.g., PCA, tICA, VAMP) 
and reweights the trajectory based on the stationary distribution derived from the transition matrix. 
This randomized clustering approach allows RiteWeight to efficiently estimate long-timescale kinetics and thermodynamics from short simulations 
— without requiring long trajectories or predefined discrete states. This code is based on the methods described in the pre-print:

Pre-print: RiteWeight: Randomized Iterative Trajectory Reweighting for Steady-State Distributions Without Discretization Error. https://arxiv.org/abs/2401.05597

---
Installation:

Install from github and update the existing conda env manually as:

git clone https://github.com/sagarkania/rite_weight.git

cd </path/to/rite_weight>

conda activate <your-env-name>

pip install .

--- 
Example:

Usage and Analysis with rite_weight Package
The example folder contains a demonstration of how to use the rite_weight package. The Jupyter notebook, example/run_riteweight.ipynb, illustrates how to build the model using data stored in the folder tests/. Additionally, the rite_weight_results_analysis.ipynb notebook provides examples of estimating pdf along a given CV.

--- 
Work in Progress:
We will be deploying features to enable kinetic estimation, such as residence time and mean first-passage time (MFPT). Stay tuned for updates.

