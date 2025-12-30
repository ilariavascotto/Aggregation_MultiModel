import os
import numpy as np
import joblib
import argparse
from scipy.stats import spearmanr as rho
from numpy.linalg import norm

import warnings
warnings.filterwarnings('ignore')

same = "_same"


parser = argparse.ArgumentParser(description="pipeline")
parser.add_argument("--dataset", type=str, default="")

args = parser.parse_args()
args.seed = np.random.randint(1)
print(args)

if args.dataset == "":
    raise ValueError("No dataset was specified")


folder = os.path.join(os.getcwd(), "Neigh", f"{args.dataset}")
folder_save = os.path.join(os.getcwd(), "Neigh",f"all{same}")

#idx_other_path = os.path.join(folder, "disagree_dict.joblib")
expl_same_path = os.path.join(folder, f"expl_disconcordant{same}.npy")
n_neigh_path = os.path.join(folder, f"n_neigh_expl_disconcordant{same}.npy")

#len_ = idx_same.shape[0]

expl = np.load(expl_same_path)
n_neigh = np.load(n_neigh_path)

print(args.dataset,  expl.shape, n_neigh.shape)

num_aggs = 3

aggregation = np.zeros((expl.shape[0], expl.shape[1], expl.shape[2], num_aggs))

for id in range(expl.shape[0]):
    expl_ = expl[id, :]

    arithmetic = np.mean(expl_, axis=2)
    quadratic = np.sqrt(expl_[:,:,0]**2 + expl_[:,:,1]**2 + expl_[:,:,2]**2)

    mix_sqrt = np.sign(arithmetic)*np.sqrt(np.abs(arithmetic*quadratic))
    
    aggregation[id, :, :, 0] = arithmetic
    aggregation[id, :, :, 1] = quadratic
    aggregation[id, :, :, 2] = mix_sqrt
    
    
np.save(os.path.join(folder_save, f"{args.dataset}_aggregations_disagree"), aggregation)
print("Saved attributions successfully")

robustness = np.zeros(shape=(expl.shape[0], num_aggs))

for i in range(expl.shape[0]):
    dim_neigh = n_neigh[i].astype(np.int32)

    if dim_neigh == 1:
        continue

    for m_ in range(num_aggs):
        rho_ = rho(aggregation[i, 0, :, m_], aggregation[i, 1:dim_neigh, :, m_], axis=1).correlation
        robustness[i, m_] = np.mean(rho_)

np.save(os.path.join(folder_save, f"{args.dataset}_robustness_disagree"),robustness)
print("Robustness saved")

