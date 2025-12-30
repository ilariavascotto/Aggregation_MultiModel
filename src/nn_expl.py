import numpy as np
import scipy
import sklearn
import os
import argparse
import joblib
import torch

import gower
from scipy.stats import spearmanr as rho

from numpy.linalg import norm
from scipy.stats import spearmanr as rho

import captum
from captum.attr import DeepLift, KernelShap, Lime, GradientShap

import utils.utils as ut
import utils.neighbourhood_generation as ng
import load.load_dataset as ds
from load import load_net

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="pipeline")
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--method", type=str, default="lime")

args = parser.parse_args()
args.seed = np.random.randint(1)
print(args)

if args.dataset == "":
    raise ValueError("No dataset was specified")

net = load_net.load_net(args.dataset)
dataset = ds.Dataset(args.dataset)

folder = os.path.join(os.getcwd(), "Neigh", f"{args.dataset}")

feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = dataset.info()
X_train_or, X_train, y_train = dataset.train()
X_val_or, X_val, y_val = dataset.validation()
X_test_or, X_test, y_test = dataset.test()
model = dataset.load_model(model_name = "model1")

path = os.path.join(folder, "neighbourhood.npy")
neigh = np.load(path)
print(neigh.shape)
neigh_size = []

if args.method == "lime":
    xaimethod = Lime(model)
elif args.method == "shap":
    xaimethod = KernelShap(model)
elif args.method == "gshap":
    xaimethod = GradientShap(model)
else:
    raise ValueError("Insert Valid Explainer") 

results = np.zeros(shape = neigh.shape)


for idx in range(neigh.shape[0]):
    x = neigh[idx, :, :]
    print(x.shape)
    target = int(np.argmax(model(X_test[idx,:]).detach().numpy()))
    x = ng.keep_neighbourhood(x, target= target, model=model, encoder = encoder)

    dim_neigh = x.shape[0]
    neigh_size.append(dim_neigh)

    if dim_neigh == 1:
        print(f"No points in the neighbourhood for test point {idx}")
        continue

    attr = xaimethod.attribute(x, target=target)
    attr = attr.detach().numpy()

    results[idx, :dim_neigh, :] = ut.reverse_encoding_neighbourhood(attr, categorical_features, categorical_names, num_ft_or=X_test_or.shape[1])

results = (results/norm(results, axis=2)[:, :, np.newaxis])
results = np.nan_to_num(results)

#np.save(os.path.join(folder, "nn_attributions"), results) #salvati a norma 1
#np.save(os.path.join(folder, "nn_neigh_size"), neigh_size) #num_points

#print("Saved attributions successfully.")

robustness = np.zeros(shape=neigh.shape[0])

for idx in range(neigh.shape[0]):
    dim_neigh = neigh_size[idx]

    if dim_neigh == 1:
        continue

    rho_ = rho(results[idx, 0, :], results[idx, 1:dim_neigh, :], axis=1).correlation
    robustness[idx]=np.mean(rho_)

np.save(os.path.join(folder, f"nn_robustness_{args.method}"), robustness)
print("Robustness saved")
    

