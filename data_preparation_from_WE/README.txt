This folder contains an exec_setup.py file that creates a 

trans_from_feat.npy, 
trans_to_feat.npy, and 
weights_in_WEweights.npy 

file based on a .h5 file of a Weighted Ensemble simulation.

Note: This workflow produces NumPy arrays of WE conformations in feature space.
To proceed with RiteWeight, you will need to apply **dimensionality reduction**
(e.g., TICA, etc.) after this stage.

1. process_ca_coord.py  
   - Performs featurization by aligning structures to reference topology file and extracting Cα (alpha carbon) xyz coordinates for each WE segment conformation.
   - This defines the feature space used in subsequent steps.
   - Modify this file to suit the molecular system and desired featurization scheme.

2. exec_setup.py  
   - Projects initial and final WE conformations into the defined feature space.
   - Extracts segment-level weights across all iterations of the Weighted Ensemble simulation.
   - Outputs `trans_from_feat.npy`: NumPy array of initial WE segment conformations.
   - Outputs `trans_to_feat.npy`: NumPy array of  WE segment conformations after dynamics run.
   - Outputs `weights_in_WEweights.npy`: NumPy array of per-segment weights for downstream reweighting.


