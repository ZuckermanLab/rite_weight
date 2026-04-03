import numpy as np

def reweights_calculate(weights_input,trans_from_to,pSSarray,learning_rate=1,smoothing_factor=0.01):
    weights_output = np.full(len(weights_input),np.nan)
    for i in range(len(pSSarray)):
        present_seg_reweight = np.where(trans_from_to[:,0]==i)
        present_seg_weights_in = weights_input[present_seg_reweight[0]] 
        if (len(present_seg_reweight[0]) > 0):
            current_sum = np.sum(present_seg_weights_in)
            scaling_factor = pSSarray[i] / current_sum
            for j in range(len(present_seg_reweight[0])):
                weights_output[present_seg_reweight[0][j]] = weights_input[present_seg_reweight[0][j]]*scaling_factor
            # optional nudge towards average weight
            idx = present_seg_reweight[0]
            cluster_mean = np.mean(weights_output[idx])
            weights_output[idx] = (
                    (1-smoothing_factor) * weights_output[idx] + smoothing_factor * cluster_mean
                    )

    sum_weights = np.sum(weights_output)
    weights_output = weights_output/sum_weights  

    
    weights_output_reduced = (weights_output * learning_rate)
    weights_new = (weights_input*(1.0-learning_rate)) + (weights_output_reduced)
    sum_weights = np.sum(weights_new)
    weights_new = weights_new/sum_weights
     
    return weights_new    
