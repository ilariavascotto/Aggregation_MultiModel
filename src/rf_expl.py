import numpy as np
import scipy
import sklearn
import os
import argparse
import joblib
import torch

import gower
from numpy.linalg import norm
from scipy.stats import spearmanr as rho

import utils.utils as ut
import utils.expl as expl
import utils.neighbourhood_generation as ng
import load.load_dataset as ds
from load import load_net

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="pipeline")
parser.add_argument("--dataset", type=str, default="")

args = parser.parse_args()
args.seed = np.random.randint(1)
print(args)

if args.dataset == "":
    raise ValueError("No dataset was specified")


dataset = ds.Dataset(args.dataset)

folder = os.path.join(os.getcwd(), "Neighbourhood", f"{args.dataset}", "test")

feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = dataset.info()
X_train_or, X_train, y_train = dataset.train()
X_val_or, X_val, y_val = dataset.validation()
X_test_or, X_test, y_test = dataset.test()

y_train_or = np.argmax(y_train.detach().numpy(), axis=1)

path = os.path.join(folder, "neighbourhood.npy")
neigh = np.load(path)

num_weak_learners = 25

# rf = RandomForestClassifier(n_estimators = num_weak_learners, max_depth=10)
# rf.fit(X_train.detach().numpy(),y_train_or)

path_save_rf = os.path.join(os.getcwd(), "models", f"{args.dataset}_rf_{num_weak_learners}.joblib")
#joblib.dump(rf, path_save_rf)
rf = joblib.load(path_save_rf)

print(rf.score(X_train.detach().numpy(), y_train_or))

neigh_size = []
results = np.zeros(shape = neigh.shape)

for idx in range(neigh.shape[0]):
    x = neigh[idx, :, :]
    if encoder!=None:
        x = encoder.transform(x).astype(np.float32)
    else:
        x = x.astype(np.float32)
        
    target = int(rf.predict(x[0, :].reshape(1,-1)))
    print(target)

    x = expl.rf_keep_neighbourhood(x, target, rf)

    dim_neigh = x.shape[0]
    neigh_size.append(dim_neigh)

    if dim_neigh ==1:
        print(f"No points in the neighbourhood for test point {idx}")

        continue

    rf_feature_used = np.zeros((neigh.shape[1], x.shape[1]))
    rf_feature_used_pos = np.zeros((neigh.shape[1], x.shape[1]))
    rf_feature_used_neg = np.zeros((neigh.shape[1], x.shape[1]))
    
    for p_id in range(dim_neigh):
        res_dict = expl.rf_path_entropy(rf, x, p_id)
        (f_p, w_p, c_p, prob_p), (f_n, w_n, c_n , prob_n) = res_dict.values()

        num = np.unique(f_p)
        for n_ in num:
            n_id = np.where(f_p==n_)[0]
            rf_feature_used_pos[p_id, n_] = np.sum(w_p[n_id])

        num = np.unique(f_n)
        for n_ in num:
            n_id = np.where(f_n==n_)[0]
            rf_feature_used_neg[p_id, n_] = np.sum(w_n[n_id])

    rf_feature_used = ((prob_n + 0.01)* rf_feature_used_pos) - ((prob_p)*rf_feature_used_neg)
    
    results[idx, : , :] = ut.reverse_encoding_neighbourhood(rf_feature_used, categorical_features, categorical_names, num_ft_or = X_test_or.shape[1])


results = (results/norm(results, axis=2)[:, :, np.newaxis])
results = np.nan_to_num(results)

np.save(os.path.join(folder, "rf_attributions_entropy"), results) #salvati a norma 1
np.save(os.path.join(folder, "rf_neigh_size_entropy"), neigh_size) #num_points

print("Saved attributions successfully.")

robustness = np.zeros(shape=neigh.shape[0])

for idx in range(neigh.shape[0]):
    dim_neigh = neigh_size[idx]

    if dim_neigh == 1:
        continue

    rho_ = rho(results[idx, 0, :], results[idx, 1:dim_neigh, :], axis=1).correlation
    robustness[idx]=np.mean(rho_)

np.save(os.path.join(folder, f"rf_robustness_entropy"), robustness)
print("Robustness saved")
