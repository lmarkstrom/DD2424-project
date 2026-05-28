import argparse

from network import Network


def lambdaSearch(lams):
    CN_params = {"f": 4, "n_f": 40, "n_s": 800}
    GD_params = {"n_cycles": 3, "n_epochs": 3, "n_hidden": 300, "k": 10, "n_batch": 100}
    LR_params = {"etas": [0.00001, 0.1]}
    results = {}

    for lam in lams:
        print(f"Testing lambda: {lam}")
        GD_params["lam"] = lam
        network = Network(LR_params, GD_params, CN_params)
        network.trainModel()
        res = network.evaluate(network.testloader)[0]
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
        help="Mode to run the script in: 'full', 'lam-search', 'train', or 'test'",
    )

    return parser.parse_args()

