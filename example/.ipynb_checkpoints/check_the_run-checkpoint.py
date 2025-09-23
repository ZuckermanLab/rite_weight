from riteweight_main import RiteWeight
import numpy as np

config = {'trans_from_Rfeat_path': './Inputs/trans_from_Rfeat.npy','trans_to_Rfeat_path': './Inputs/trans_to_Rfeat.npy'}

my_model = RiteWeight(config)

initial_weights = np.load('./Inputs/weights_in_Uniform_microstate_wise.npy')
iteration = 1000
clusters = 500
weight_freq = 100
my_model.run_RiteWeight_iter(initial_weights,clusters,iteration,weight_freq)

