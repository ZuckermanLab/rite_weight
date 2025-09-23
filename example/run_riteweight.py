from rite_weight import rite_weight 
import numpy as np

config = {'trans_from_Rfeat_path': './Inputs/trans_from_Rfeat.npy','trans_to_Rfeat_path': './Inputs/trans_to_Rfeat.npy'} #Creating dict
my_model = rite_weight.rite_weight_model(config) # Initialize the rite_weight calss


initial_weights = np.load('./Inputs/weights_in_uniform_microstate_wise.npy')
total_iter = 10000
clusters = 10
weight_out_freq = 10
my_model.rite_weight_iter(initial_weights,clusters,total_iter,weight_out_freq)

