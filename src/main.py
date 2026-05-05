import os
import warnings
import argparse
import numpy as np
from network import Network

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

def trainFullNetwork():
    CN_params = {'f': 4, 'n_f': 40}
    GD_params = {'n_cycles': 3, 'n_epochs': 4, 'm': 400, 'k': 10, 'n_batch': 100, 'lam': 0.0004}
    LR_params = {'etas': [0.00001, 0.1], 'n_s': None}
    
    network = Network(LR_params, GD_params, CN_params)
    
    network.loadData()
    network.train(debug=True)
    network.evaluate()
    
def argParser():
    parser = argparse.ArgumentParser(description="CNN Project Runner")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        help="Mode to run the script in: 'full', 'train', or 'test'"
    )

    return parser.parse_args()

def main():
    np.random.seed(42) # TODO: Remove for more randomness, which can sometimes give better results
    
    args = argParser()
    
    if args.mode == "full":
        trainFullNetwork()
    else:
        print(f"Mode '{args.mode}' is not implemented yet.")
    
if __name__ == "__main__":
    main()