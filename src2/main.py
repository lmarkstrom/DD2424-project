import os
import warnings
import numpy as np
from network import Network
import torch
from mainHelpers import argParser, lambdaSearch

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

def trainNet():
    CN_params = {'f': 4, 'n_f': 40, 'n_s': 800}
    GD_params = {'n_cycles': 3, 'n_epochs': 4, 'n_hidden': 300, 'k': 10, 'n_batch': 100, 'lam': 0.0025}
    CN_params = {
        'l_patchify': {'f': 2, 's': 2, 'n_f': 32},
        'l_vgg1': {'f': 3, 's': 1, 'n_f': 32},
        'l_fc1': {'in': 32 * 8 * 8, 'out': GD_params['n_hidden']},
        'l_fc2': {'in': GD_params['n_hidden'], 'out': GD_params['k']}
    }
    LR_params = {'etas': [1e-5, 1e-1]}
    
    x = torch.rand((1, 3, 32, 32))
    network = Network(LR_params, GD_params, CN_params)
    x = network.forward(x)
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