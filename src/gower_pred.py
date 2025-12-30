import os
import numpy as np
import pickle
import joblib
import load.load_dataset as ds
import torch


for dataset_name in ["adult", "bank"]:
    print(dataset_name)
    path_knn = os.path.join(os.getcwd(), "models", f"{dataset_name}_knn_15.joblib")
    
    dataset = ds.Dataset(dataset_name=dataset_name)
    knn = joblib.load(path_knn)


    feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = dataset.info()
    Xtrain_or, Xtrain, ytrain = dataset.train()
    Xvalid_or, Xvalid, yvalid = dataset.validation()
    Xtest_or, Xtest, ytest = dataset.test()

    folder = os.path.join(os.getcwd(), "Neighbourhood", dataset_name, "test")

    path_gower_pickle = os.path.join(folder, "gower_test_neigh.pkl")
    path_gower_npy = os.path.join(folder, "gower_test_neigh.npy")
    
    try:
        gower_test = np.load(path_gower_npy)
    except:
        with open(path_gower_pickle, "rb") as f:
            gower_test = pickle.load(f)
    
    print(gower_test.shape)
    #gower_test = gower_test.reshape(Xtest_or.shape[0], 101, gower_test.shape[1])
    
    y_hat_knn = knn.predict(gower_test)
    print(y_hat_knn.shape)

    y_hat_knn = y_hat_knn.reshape(Xtest_or.shape[0], 101)
    print(y_hat_knn.shape)
    np.save(os.path.join(folder, "knn_gower_pred_neigh"), y_hat_knn)