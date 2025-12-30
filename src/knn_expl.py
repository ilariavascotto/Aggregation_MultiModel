import numpy as np
import scipy
import sklearn
import os
import argparse
import joblib
import torch

import pickle
import gower
from scipy.stats import spearmanr as rho

from numpy.linalg import norm
from scipy.stats import spearmanr as rho

import utils.utils as ut
import utils.expl as expl
import utils.neighbourhood_generation as ng
import utils.condensation as cond
import load.load_dataset as ds
from load import load_net

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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

folder = os.path.join(os.getcwd(), "Neigh", f"{args.dataset}") 

feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = dataset.info()
X_train_or, X_train, y_train = dataset.train()
X_val_or, X_val, y_val = dataset.validation()
X_test_or, X_test, y_test = dataset.test()
num = 100

bool_vars = [0 if i not in categorical_features else 1 for i in range(X_val_or.shape[1])]

y_train_or = np.argmax(y_train.detach().numpy(), axis=1)


path_train = os.path.join(folder, "gower_train.npy")
path_test = os.path.join(folder, "gower_test_neigh.npy")
path_test_pickle = os.path.join(folder, "gower_test_neigh.pkl")

gower_train = np.load(path_train)

try:
    gower_test_neigh = np.load(path_test)
except:
    with open(path_test_pickle, "rb") as f:
        gower_test_neigh = pickle.load(f)

try:
    gower_test = gower_test_neigh.reshape(X_test_or.shape[0], num +1, X_train_or.shape[0])
except:
    print("Cannot reshape test matrix")

print(gower_train.shape, gower_test.shape)


path_neighbourhood = os.path.join(folder, "neighbourhood.npy")
neigh = np.load(path_neighbourhood)
print(neigh.shape)

if args.dataset in ["adult", "bank"]:
    k_nn = 15
else:
    k_nn = 5
# knn = KNeighborsClassifier(n_neighbors=k_nn, metric='precomputed').fit(gower_train, y_train_or)
# y_hat = knn.predict(gower_train)

# print(np.mean(y_hat==y_train_or))

path_save_knn = os.path.join(os.getcwd(), "models", f"{args.dataset}_knn_{k_nn}.joblib")
# joblib.dump(knn, path_save_knn)

knn = joblib.load(path_save_knn)


neigh_size = []
results = np.zeros(shape = neigh.shape)

num_ranges, num_max = expl.numerical_range_max(np.concatenate([X_train_or, X_val_or, X_test_or]), bool_vars)


print("check one")
for idx in range(neigh.shape[0]):
    x = gower_test[idx, :, :]

    y_pred_class = knn.predict(x)
    y_pred = np.where(y_pred_class == y_pred_class[0])[0]
    
    X_knn_same = np.zeros(shape = (x.shape[0], k_nn, X_test_or.shape[1]))
    X_knn_other = np.zeros(shape = (x.shape[0], k_nn, X_test_or.shape[1]))   

    len_same = np.zeros(shape = x.shape[0])


    n_neigh = len(y_pred)
    neigh_size.append(n_neigh)

    # if n_neigh != 1:
    #     if y_pred_class[0] == 0:
    #         idx00 = knn_0.kneighbors(x[y_pred][:,y_cl_0], return_distance=False)
    #         idx01 = knn_1.kneighbors(x[y_pred][:,y_cl_1], return_distance=False)

    #         X_knn_same[y_pred, :, :] = X_train_or[idx00, :]
    #         X_knn_other[y_pred, :, :] = X_train_or[idx01, :]

    #     if y_pred_class[0]==1:
    #         idx10 = knn_0.kneighbors(x[y_pred][:,y_cl_0], return_distance=False)
    #         idx11 = knn_1.kneighbors(x[y_pred][:,y_cl_1], return_distance=False)        

    #         X_knn_same[y_pred, :, :] = X_train_or[idx11, :]
    #         X_knn_other[y_pred, :, :] = X_train_or[idx10, :]

    if n_neigh != 1:
        idx_ = knn.kneighbors(x, return_distance = False)
        y_neighbours = y_train_or[idx_]

        for el in range(x.shape[0]):
            idx_el =  idx_[el]
            y_neigh_el = y_neighbours[el]
        
            same_ = np.where(y_neigh_el == y_pred_class[el])[0]
            other_ = np.where(y_neigh_el != y_pred_class[el])[0]

            len_same[el] = len(same_)

            idx_same = idx_el[same_]
            idx_other = idx_el[other_]

            X_knn_same[el, :len(same_)] = X_train_or[idx_same]
            X_knn_other[el, :len(other_)] = X_train_or[idx_other]
            

    neigh_x = neigh[idx, :, :]
            
    gower_same = expl.gower_feature_wise_test(neigh_x, X_knn_same, bool_vars, num_ranges, num_max)
    gower_other = expl.gower_feature_wise_test(neigh_x, X_knn_other, bool_vars, num_ranges, num_max)

    gower_same_mean = np.sum(gower_same, axis=1)
    gower_other_mean = np.sum(gower_other, axis=1)

    pr_same = len_same/k_nn
    pr_other = 1-len_same

    results[idx, : , :] = ((pr_other[:, np.newaxis] + 0.01)*gower_same_mean) - (pr_same[:, np.newaxis]* gower_other_mean)

#QUESTO OK
print("check due", np.mean(neigh_size))
results = (results/norm(results, axis=2)[:, :, np.newaxis])
results = np.nan_to_num(results)

np.save(os.path.join(folder, f"knn_attributions_same"), results) #salvati a norma 1
np.save(os.path.join(folder, f"knn_neigh_size_same"), neigh_size)
print("Saved attributions successfully.")

robustness = np.zeros(shape=X_test_or.shape[0])

for idx in range(neigh.shape[0]):

    rho_ = rho(results[idx, 0, :], results[idx, 1:, :], axis=1).correlation
    robustness[idx]=np.mean(rho_)

np.save(os.path.join(folder, f"knn_robustness_same"), robustness)
print("Robustness saved")
    
