import numpy as np
import h5py 
from rite_weight.clustering_utils import RandomClustering
import rite_weight.transition_matrix_utils as tm
from rite_weight.reweighting_utils import reweights_calculate

class rite_weight_model:
    def __init__(self, config):
        # Load input arrays and other parameters
        self.trans_from_feat = np.load(config['trans_from_Rfeat_path'])
        self.trans_to_feat = np.load(config['trans_to_Rfeat_path'])
        self.thero_weight_ref = 0.0  #Change to very small value bcz of which NAN could be there in Tmatrix
        
        # Round to 3 decimals
        trans_from_feat_rounded = np.round(self.trans_from_feat, decimals=3)
        # Extract unique rows and their indices
        unique_rows, unique_indices = np.unique(trans_from_feat_rounded, axis=0, return_index=True)
        # Store for use in other methods
        self.ref_segs_feat = unique_rows
        self.ref_segs_indices = unique_indices
    
        
    def rite_weight_iter(self,weights_started,clusters,total_iterations,weightout_fq):
        self.total_clusters = clusters
        self.clusterer = RandomClustering(self.trans_from_feat,self.trans_to_feat,self.total_clusters)
        
        total_segs = len(self.trans_from_feat)
        with h5py.File('Reweights.h5', 'w') as f:
            dset1 = f.create_dataset('weights_out', (total_iterations//weightout_fq, total_segs))
        
        weights_in = weights_started
        for reweight_iteration in range(total_iterations):
            pSS_min = 0.0
            while pSS_min <= 0:
                trans_index_pair = self.clusterer.get_random_clusters_cdist(self.ref_segs_feat)
                Tmatrix = tm.get_Tmatrix(weights_in,trans_index_pair) 
                pSS = tm.get_steady_state(Tmatrix)
                pSS_min = np.min(pSS)  
            # Reweight traj 
            weights_out = reweights_calculate(weights_in,trans_index_pair,pSS)   
            weights_in = weights_out 
            
            if ( ((reweight_iteration%weightout_fq)==0) ):
                with h5py.File('Reweights.h5', 'a') as f:
                     dset = f['weights_out']
                     dset[reweight_iteration // weightout_fq,:] = weights_out 
            
            #Remove low weights Ref Seg 
            neg_weight_seg = np.where(weights_in[:]<= self.thero_weight_ref)    
            if (len(neg_weight_seg[0]) > 0):
                neg_weight_seg_array = np.array(neg_weight_seg[0])
                for i_to_remove in range(len(neg_weight_seg_array)):
                    ref_to_remove_indices = None
                    ref_to_remove_indices = np.where(ref_segs_indices==neg_weight_seg_array[i_to_remove])
                    self.ref_segs_indices = np.delete(self.ref_segs_indices, ref_to_remove_indices[0], axis=0)
                    self.ref_segs_feat = np.delete(self.ref_segs_feat, ref_to_remove_indices[0], axis=0)
            
            
