import os
import numpy as np
import pickle
import joblib
import load.load_dataset as ds
import torch

same = "_same"

for dataset_name in ["adult", "bank"]: #["cancer", "wine", "heloc"]:
    print(dataset_name)
    #path_knn = os.path.join(os.getcwd(), "models", f"{dataset_name}_knn_5.joblib")
    path_rf = os.path.join(os.getcwd(), "models", f"{dataset_name}_rf_25.joblib")
    path_nn = os.path.join(os.getcwd(), "models", f"{dataset_name}_model1.pt")

    dataset = ds.Dataset(dataset_name=dataset_name)
    nn = dataset.load_model("model1")
    #knn = joblib.load(path_knn)
    rf = joblib.load(path_rf)

    feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = dataset.info()
    Xtrain_or, Xtrain, ytrain = dataset.train()
    Xvalid_or, Xvalid, yvalid = dataset.validation()
    Xtest_or, Xtest, ytest = dataset.test()

    folder = os.path.join(os.getcwd(), "Neigh", dataset_name)
    neigh_path = os.path.join(folder, "neighbourhood.npy")
    neigh = np.load(neigh_path)

    # path_gower_pickle = os.path.join(folder, "gower_test_neigh.pkl")
    # path_gower_npy = os.path.join(folder, "gower_test_neigh.npy")
    
    # try:
    #     gower_test = np.load(path_gower_npy)
    # except:
    #     with open(path_gower_pickle, "rb") as f:
    #         gower_test = pickle.load(f)
    
    # gower_test = gower_test.reshape(Xtest_or.shape[0], 101, gower_test.shape[1])
    
    #y_hat_knn = knn.predict(gower_test[:,0,:])
    y_hat_rf = rf.predict(Xtest.detach().numpy())
    y_hat_nn = np.argmax(nn(Xtest).detach().numpy(), axis=1)

    # idx = np.where(np.logical_and(y_hat_knn==y_hat_rf, y_hat_rf==y_hat_nn))[0] #punti su cui i tre modelli concordano di previsione
    # print(len(idx))
    # np.save(os.path.join(folder, "model_concordance_idx_test"), idx)



    knn_pred_all = np.load(os.path.join(folder, "knn_gower_pred_neigh.npy"))
    
    
    idx = np.load(os.path.join(folder, "model_concordance_idx_test.npy"))


    path_knn_expl = os.path.join(folder, f"knn_attributions{same}.npy")
    path_rf_expl = os.path.join(folder, "rf_attributions.npy")
    path_nn_expl = os.path.join(folder, "nn_attributions.npy")

    knn_expl = np.load(path_knn_expl)
    rf_expl = np.load(path_rf_expl)
    nn_expl = np.load(path_nn_expl)

    print(knn_expl.shape, rf_expl.shape, nn_expl.shape)
    a, b, c = knn_expl.shape
    expl = np.zeros((a, b, c, 3))
    n_neigh = np.zeros(a)

##########################################################################################################################################

    for id in range(neigh.shape[0]):
        if id in idx:
            x = neigh[id, :, :]
            if encoder!=None:
                x = encoder.transform(x).astype(np.float32)
            else:
                x = x.astype(np.float32)
            # x_gower = gower_test[id, :, :]

            try:
                x_tensor = torch.Tensor(x)
            except:
                x_tensor = torch.Tensor(x.toarray())


            y_hat_knn = knn_pred_all[id]  #knn.predict(x_gower)
            y_hat_rf = rf.predict(x)
            y_hat_nn = np.argmax(nn(x_tensor).detach().numpy(), axis=1)

            idx_same= np.where(np.logical_and(y_hat_knn==y_hat_rf, y_hat_rf==y_hat_nn)==True)[0]

            idx_knn = np.where(y_hat_knn == y_hat_knn[0])[0]
            idx_rf= np.where(y_hat_rf == y_hat_rf[0])[0]
            idx_nn = np.where(y_hat_nn == y_hat_nn[0])[0]

            # knn_list = [i for i, el in enumerate(idx_knn) if el in idx_same]
            # rf_list = [i for i, el in enumerate(idx_rf) if el in idx_same]
            # nn_list = [i for i, el in enumerate(idx_nn) if el in idx_same]

            all_list = list(set.intersection(*map(set,[idx_same, idx_knn, idx_rf, idx_nn])))


            l = len(all_list)
            print(l)
            expl[id, :l, :, 0] = knn_expl[id, all_list, :]
            expl[id, :l, :, 1] = rf_expl[id, all_list, :]
            expl[id, :l, :, 2] = nn_expl[id, all_list, :]

            n_neigh[id] = l



    np.save(os.path.join(folder, f"expl_concordant{same}"), expl)
    np.save(os.path.join(folder, f"n_neigh_expl_concordant{same}"), n_neigh)


        
    