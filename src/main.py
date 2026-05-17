import os
import warnings
import numpy as np
from network import Network
from mainHelpers import argParser, lambdaSearch

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

def trainNet():
    CN_params = {'f': 4, 'n_f': 40, 'n_s': 800}
    GD_params = {'n_cycles': 1, 'n_epochs': 60, 'n_hidden': 300, 'k': 10, 'n_batch': 100, 'img_size': 32, 'lam': 0.0001}
    CN_params = {
        'l_patchify': {'f': 2, 's': 2, 'n_f': 64},
        'l_vgg1': {'f': 3, 's': 1, 'n_f': 64},
        'l_vgg2': {'f': 3, 's': 1, 'n_f': 128},
        'l_vgg3': {'f': 3, 's': 1, 'n_f': 256},
        'l_fc1': {'in': 256 * 4 * 4, 'out': GD_params['n_hidden']},
        'l_fc2': {'in': GD_params['n_hidden'], 'out': GD_params['k']}
    }
    LR_params = {
        'eta': 1e-2,
        'scheduler': True,
        'scheduler_type': "cosine"} # "step", "cosine", ...
    RE_params = {
        'dropout': True, 'dropout_rates': [0.3, 0.4, 0.5, 0.5], 
        'augementation': True, 'flip_prob': 0.5, 'shift_max': 0.1,
        'label_smoothing': True, 'smoothing_factor': 0.1}
    
    network = Network(LR_params, GD_params, CN_params, RE_params)
    network.trainModel(plot=True)
    res = network.evaluate(network.testloader)[0]
    print(f'Final Test Accuracy: {res}')



def main():
    np.random.seed(42) # TODO: Remove for more randomness, which can sometimes give better results
    
    args = argParser()
    
    if args.mode == "full":
        trainNet()
    elif args.mode == "lam-search":
        lambdaSearch([0.0001, 0.00025, 0.0005, 0.001]) # Lambdas = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01]
    else:
        print(f"Mode '{args.mode}' is not implemented yet.")
    
if __name__ == "__main__":
    main()