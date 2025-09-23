import numpy as np
import random
from scipy.spatial import distance 

class RandomClustering:
    def __init__(self, trans_from_feat, trans_to_feat, total_clusters):
        self.trans_from_feat = trans_from_feat
        self.trans_to_feat = trans_to_feat
        self.total_clusters = total_clusters

    def get_random_clusters_cdist(self, ref_segs_feat):
        idx = np.random.choice(ref_segs_feat.shape[0], size=self.total_clusters, replace=False)
        cc_feats = ref_segs_feat[idx]
        trans_from_dis_array = distance.cdist(self.trans_from_feat, cc_feats, 'euclidean')
        trans_to_dis_array = distance.cdist(self.trans_to_feat, cc_feats, 'euclidean')

        trans_from_index = np.argmin(trans_from_dis_array,axis=1)
        trans_to_index = np.argmin(trans_to_dis_array,axis=1)
        trans_pair = np.column_stack((trans_from_index,trans_to_index))
        
        return trans_pair


        