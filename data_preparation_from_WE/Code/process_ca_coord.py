import mdtraj as md
import numpy as np
from itertools import combinations

topology  = "path_to_topology_file.pdb"

def processCoordinates(coords):
    
    reference_structure = md.load(topology)
    xt = md.Trajectory(xyz=coords, topology=reference_structure.topology)

    #  C-alpha atoms indices 
    ca_ind = reference_structure.topology.select("name CA")

    xt.superpose(reference_structure,frame=0,atom_indices=ca_ind,ref_atom_indices=ca_ind)

    # extract CA coordinates
    ca_xyz = xt.xyz[:, ca_ind, :]   # nm
    ca_xyz = ca_xyz.reshape(ca_xyz.shape[0], -1)
    
    # convert to Angstrom if desired
    ca_xyz = ca_xyz * 10.0

    return ca_xyz