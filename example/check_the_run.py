from rite_weight import rite_weight 
import numpy as np

config = {'trans_from_Rfeat_path': './Inputs/trans_from_Rfeat.npy','trans_to_Rfeat_path': './Inputs/trans_to_Rfeat.npy'}

my_model = rite_weight.rite_weight_model(config)

initial_weights = np.load('./Inputs/weights_in_uniform_microstate_wise.npy')
iteration = 10
clusters = 500
weight_freq = 1
my_model.rite_weight_iter(initial_weights,clusters,iteration,weight_freq)

