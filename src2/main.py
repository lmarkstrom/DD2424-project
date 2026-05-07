import os
import warnings
import argparse
import numpy as np
from network import Network
import torch

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

def trainNet():
    CN_params = {'f': 4, 'n_f': 40, 'n_s': 800}
    GD_params = {'n_cycles': 3, 'n_epochs': 4, 'n_hidden': 300, 'k': 10, 'n_batch': 100, 'lam': 0.0025}
    LR_params = {'etas': [0.00001, 0.1]}
    
    x = torch.rand((1, 3, 32, 32))
    network = Network(LR_params, GD_params, CN_params)
    x = network.forward(x)
    network.trainModel()
    res = network.evaluate()
    print(res)

def lambdaSearch(lams):
    CN_params = {'f': 4, 'n_f': 40, 'n_s': 800}
    GD_params = {'n_cycles': 3, 'n_epochs': 3, 'n_hidden': 300, 'k': 10, 'n_batch': 100}
    LR_params = {'etas': [0.00001, 0.1]}
    results = {}
    
    for lam in lams:
        print(f"Testing lambda: {lam}")
        GD_params['lam'] = lam
        network = Network(LR_params, GD_params, CN_params)
        network.trainModel()
        res = network.evaluate()
        results[lam] = res
    
    print("Lambda Search Results:")
    for lam, res in results.items():
        print(f"Lambda: {lam}, Accuracy: {res}")
    print("Best Lambda:", max(results, key=lambda k: results[k]))

def argParser():
    parser = argparse.ArgumentParser(description="CNN Project Runner")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        help="Mode to run the script in: 'full', 'lam-search', 'train', or 'test'"
    )

    return parser.parse_args()

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