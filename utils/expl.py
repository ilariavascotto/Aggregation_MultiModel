import numpy as np
import sklearn.ensemble


def rf_path_entropy(rf, X, point_id):
 
    f_pos, w_pos = [], []
    f_neg, w_neg = [], []
    c_pos, c_neg = 0, 0

    pred = rf.predict(X[point_id,:].reshape(1,-1))
    pred_prob = rf.predict_proba(X[point_id, :].reshape(1,-1))[0]

    for est in rf.estimators_:        
        tree = est.tree_
        assert tree.value.shape[1] == 1

        if np.argmax(tree.predict(X[point_id, :].reshape(1,-1))) == pred:
            features, weight = f_pos, w_pos
            c_pos += 1
        else:
            features, weight = f_neg, w_neg
            c_neg += 1
        
        feature = tree.feature
        # threshold = tree.threshold
        node_indicator = tree.decision_path(X)
        leaf_id = tree.apply(X)

        gini = tree.impurity

        node_index = node_indicator.indices[node_indicator.indptr[point_id]:node_indicator.indptr[point_id+1]]

        for id, node_id in enumerate(node_index):
            if leaf_id[point_id] == node_id:
                pass
            else:
                features.append(feature[node_id])
                weight.append(gini[node_id])
        
    pred = np.int32(pred)
    prob_p = pred_prob[pred]
    prob_n = pred_prob[1-np.abs(pred)]


    res_dict = {"pos": (f_pos, np.array(w_pos), c_pos, prob_p), 
               "neg": (f_neg, np.array(w_neg), c_neg, prob_n)}
    return res_dict


def rf_keep_neighbourhood(neigh, target, rf):
    keep = np.where(rf.predict(neigh) == target)[0]

    return neigh[keep]



def numerical_range_max(X, bool_vars):
    idx = np.where(np.array(bool_vars)==0)[0]
    X = X[:, idx]
    row, cols = X.shape

    num_ranges = np.zeros(cols)
    num_max = np.zeros(cols)    

    for col in range(cols):
        col_array = X[:, col].astype(np.float32)
        max = np.nanmax(col_array)
        min = np.nanmin(col_array)

        if np.isnan(max):
            max = 0.0
        if np.isnan(min):
            min = 0.0
        
        num_max[col] = max
        num_ranges[col] = np.abs(1 - min/max) if (max != 0) else 0.0
    
    return num_ranges, num_max

def gower_feature_wise_test(X_test_or, X_knn, bool_vars, num_ranges, num_max):
    if (num_ranges==0.0).any():
        raise ValueError
    if (num_max==0.0).any():
        raise ValueError
    
    X_test = X_test_or[:, np.newaxis, :]
    idx_cat = np.where(np.array(bool_vars)==1)[0]
    idx_num = np.where(np.array(bool_vars)==0)[0]   

    X_test[:, :, idx_num] = np.divide(X_test[:, :, idx_num], num_max)
    X_knn[:, :, idx_num] = np.divide(X_knn[:, :, idx_num], num_max)

    X_num = np.absolute(X_knn[:, :, idx_num] - X_test[:, :, idx_num])
    X_num = np.divide(X_num, num_ranges)

    X_cat = (X_knn[:, :, idx_cat] == X_test[:, :, idx_cat]).astype(np.int32)

    gower_feature_wise = np.zeros(X_knn.shape)
    gower_feature_wise[:, :, idx_num] = X_num
    gower_feature_wise[:, :, idx_cat] = X_cat

    gower_feature_wise = np.divide(gower_feature_wise, len(bool_vars))

    return gower_feature_wise