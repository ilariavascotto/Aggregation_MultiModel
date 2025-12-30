import numpy as np
import sklearn

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn_extra.cluster import KMedoids

def grow_cluster(reference, p, labels, neighbours_list, C, idx):
    reference[p] = C

    i = 0
    while i < len(neighbours_list):
        neighbour = neighbours_list[i]
        neighs = idx[neighbour, :]

        for neigh in neighs:
            if reference[neigh] == -1 :
                reference[neigh] = C
                
                new_neighbors = idx[neigh, :]
                if np.mean(labels[new_neighbors]==labels[p])== 1:
                    neighbours_list += list(new_neighbors)
        i += 1

    return

def find_medoids(reference, distances, max_size, min_count):
    idx, count = np.unique(reference[reference>0], return_counts = True) #remove unassigned points (-1) or diff neighbours points (0)
    count_cluster = np.where(count > min_count)[0] #remove clusters with only one point

    clusters = idx[count_cluster]
    cl_size = count[count_cluster]

    k_cl = np.ceil(cl_size/max_size).astype(np.int64)
    n_cl = np.sum(k_cl)

    indices = np.zeros(shape=n_cl)

    acc_ = 0

    for c_id, c in enumerate(clusters):
        idx_cl = np.where(reference==c)[0]

        k_ = k_cl[c_id]
        dist_cl = distances[idx_cl][:, idx_cl]

        kmedoids = KMedoids(n_clusters= k_, metric = 'precomputed')
        kmedoids.fit(dist_cl)

        medoid_indices = kmedoids.medoid_indices_

        indices[acc_ : acc_ + k_] = idx_cl[medoid_indices]
        acc_ += k_

    return indices.astype(np.int64) #medoids


def condensation(knn, k, labels, distances, max_size = 10, min_count = 1):
    dist, idx = knn.kneighbors(n_neighbors = k+1, X = distances)

    reference = np.full(fill_value = 0, shape = labels.shape[0])

    same_neigh_idx = np.where(np.mean(labels[:, np.newaxis]==labels[idx], axis=1)==1)[0] #consider only points already in agreement with neighbours
    reference[same_neigh_idx] = -1

    C = 1
    for p in same_neigh_idx:
        if reference[p] != -1:
            continue

        neighbours = idx[p, :]
        grow_cluster(reference, p, labels, list(neighbours), C, idx)
        C += 1

    medoid_indices = find_medoids(reference, distances, max_size = max_size, min_count = min_count)

    return medoid_indices

