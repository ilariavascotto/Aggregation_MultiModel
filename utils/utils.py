import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import warnings
warnings.filterwarnings('ignore')

def reverse_encoding_neighbourhood(data, categorical_features, categorical_names, num_ft_or):
    try:
        data = data.detach().numpy()
    except:
        pass

    num_points, ft = data.shape

    lens = [len(categorical_names[i]) for i in categorical_names.keys()]
    acc = [int(np.sum(lens[:i])) for i in range(len(lens)+1)]

    numerical_features = [i for i in range(num_ft_or) if i not in categorical_features]

    X = np.zeros(shape = (num_points, num_ft_or))

    for idx, i in enumerate(range(len(acc)-1)):
        i_ = categorical_features[idx]
        X[:, i_] = np.sum(data[:, acc[i]:acc[i+1]], axis=1)

    for idx, j in enumerate(range(ft-acc[-1])):
        j_ = numerical_features[idx]
        X[:,j_] = data[:, j+acc[-1]]
    return X
