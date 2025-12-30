import numpy as np
import gower
from sklearn.neighbors import NearestNeighbors
from kmedoids import KMedoids

def km_gower(X, n_clust, bool_vars):
    dist = gower.gower_matrix(X, cat_features = bool_vars)
    
    kmedoids = KMedoids(n_clusters = n_clust, metric='precomputed')
    kmedoids.fit(dist)

    medoid_indices = kmedoids.medoid_indices_
    cluster_centers = X[medoid_indices]
    labels = kmedoids.labels_

    return kmedoids, medoid_indices, cluster_centers, labels

def knn_overall(cluster_centers, medoid_indices, n_neigh,  bool_vars):
    dist_centers = gower.gower_matrix(cluster_centers, cat_features = bool_vars)
    knn = NearestNeighbors(metric = 'precomputed').fit(dist_centers)
    _knn_ = knn.kneighbors(dist_centers, n_neighbors=n_neigh +1, return_distance=False)[:, 1:]

    return _knn_

def km_predict(X, cluster_centers, bool_vars):
    n = X.shape[0]

    X_new = np.concatenate([X, cluster_centers])

    dist = gower.gower_matrix(X_new, cat_features = bool_vars)
    dist = dist[:n, n:]

    return np.argmin(dist, axis=1)



## to deal only with categorical variables (mushroom dataset) we can use the manhattan distance instead

def km_manhattan(X, n_clust):
    kmedoids = KMedoids(n_clusters = n_clust, init='random', metric = 'manhattan')
    kmedoids.fit(X)
    
    cluster_centers = kmedoids.cluster_centers_
    labels = kmedoids.labels_

    return kmedoids, cluster_centers, labels

def knn_overall_manhattan(cluster_centers, n_neigh):
    knn = NearestNeighbors(metric = 'manhattan').fit(cluster_centers)
    knn_overall = knn.kneighbors(cluster_centers, n_neighbors=n_neigh +1, return_distance=False)

    return knn_overall[:,1:]

def km_predict_manhattan(X, kmedoids):
    labels = kmedoids.labels_[kmedoids.predict(X)]
    return labels

