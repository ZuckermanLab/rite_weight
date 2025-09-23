import numpy as np

def reweights_calculate(weights_input,trans_from_to,pSSarray):
    weights_output = np.full(len(weights_input),np.nan)
    for i in range(len(pSSarray)):
        present_seg_reweight = np.where(trans_from_to[:,0]==i)
        present_seg_weights_in = weights_input[present_seg_reweight[0]] 
        if (len(present_seg_reweight[0]) > 0):
            current_sum = np.sum(present_seg_weights_in)
            scaling_factor = pSSarray[i] / current_sum
            for j in range(len(present_seg_reweight[0])):
                weights_output[present_seg_reweight[0][j]] = weights_input[present_seg_reweight[0][j]]*scaling_factor
    sum_weights = np.sum(weights_output)
    weights_output = weights_output/sum_weights  

    gamma = 1.0                                                     #Change to variable in next update
    weights_output_reduced = (weights_output * gamma)
    weights_new = (weights_input*(1.0-gamma)) + (weights_output_reduced)
    sum_weights = np.sum(weights_new)
    weights_new = weights_new/sum_weights
     
    return weights_new    
