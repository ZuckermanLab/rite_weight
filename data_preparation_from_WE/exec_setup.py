import numpy as np
import pickle
import h5py
import sys
sys.path.append("./Code")
import process_ca_coord #having this function will do featurization for eg: here the function returns c_alpha xyz coordinates. 



folder='./folder_path_with_h5/' 
dataIn = h5py.File(folder+'path_to_h5_file.h5', "r")


# -----------------------------------------
# determine iterations
# -----------------------------------------

iters = list(dataIn["iterations"].keys())

maxIter = max(int(k.split("_")[1]) for k in iters)

print("Max iteration:", maxIter)

first_iter = 1
iters_to_use = range(first_iter, maxIter)

all_iter_process_coordinates = None


for iteration in iters_to_use:
    dsetName = "/iterations/iter_%08d/auxdata/coord" % int(iteration)   #the path in west.h5 where auxdata is stored
    dset = dataIn[dsetName]
    dset_array = np.array(dset)
    iter_coords_transition_from = dset_array[:,0,:,:]
    iter_process_coords_transition_from = process_ca_coord.processCoordinates(iter_coords_transition_from)
    if all_iter_process_coordinates is None:
        all_iter_process_coordinates = iter_process_coords_transition_from
    else:
        all_iter_process_coordinates = np.vstack((all_iter_process_coordinates, iter_process_coords_transition_from))

    
print(all_iter_process_coordinates.shape)
np.save(folder+'trans_from_feat.npy',all_iter_process_coordinates) #Units are of coord unit.


all_iter_process_coordinates = None


for iteration in iters_to_use:
    dsetName = "/iterations/iter_%08d/auxdata/coord" % int(iteration)
    dset = dataIn[dsetName]
    dset_array = np.array(dset)
    iter_coords_transition_to = dset_array[:,-1,:,:]
    iter_process_coords_transition_to = process_ca_coord.processCoordinates(iter_coords_transition_to)
    if all_iter_process_coordinates is None:
        all_iter_process_coordinates = iter_process_coords_transition_to
    else:
        all_iter_process_coordinates = np.vstack((all_iter_process_coordinates, iter_process_coords_transition_to))

    
print(all_iter_process_coordinates.shape)
np.save(folder+'trans_to_feat.npy',all_iter_process_coordinates)

all_iter_seg_weights = None
for iteration in iters_to_use:
    dsetName = "/iterations/iter_%08d/seg_index" % int(iteration)
    dsetin = dataIn[dsetName]
    weights = None
    weights = dsetin["weight"]
    weights = weights.reshape(len(weights),1)
    if all_iter_seg_weights is None:
        all_iter_seg_weights = weights
    else:
        all_iter_seg_weights = np.vstack((all_iter_seg_weights, weights))
        

sum_weights_all_iter = np.sum(all_iter_seg_weights[:,0])
print(sum_weights_all_iter)
all_iter_seg_weights = all_iter_seg_weights/sum_weights_all_iter
all_iter_seg_weights = all_iter_seg_weights.reshape(len(all_iter_seg_weights))
print(np.sum(all_iter_seg_weights))
print(all_iter_seg_weights.shape)
np.save(folder+'weights_in_WEweights.npy',all_iter_seg_weights)