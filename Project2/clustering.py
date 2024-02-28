import numpy as np
from scipy.spatial import distance


def nearest_center(centers, X): #find which center is closest to a given point
    distances = np.zeros([len(centers), 2])

    for i in range(len(centers)):
        distances[i][0] = distance.euclidean(centers[i], X)
        distances[i][1] = i


    distances = distances[np.argsort(distances[:, 0])]


    return int(distances[0][1])




def K_Means(X,K,mu):
    if hasattr(X[0], "__len__"):
        init_centers = np.zeros([K, len(X[0])])
    else:
        init_centers = np.zeros(K)

    if len(mu) == 0:
        center_indices = np.random.choice(len(X), K, replace = False)
        for i in range(K):
            init_centers[i] = X[center_indices[i]]
    else:
        for i in range(K):
            init_centers[i] = mu[i]

    final_centers = init_centers
    if hasattr(X[0], "__len__"):
        init_centers = np.zeros([K, len(X[0])])
    else:
        init_centers = np.zeros(K)
        
    while (np.array_equal(init_centers, final_centers) == False):
        init_centers = final_centers

        if hasattr(X[0], "__len__"):
            final_centers = np.zeros([K, len(X[0])])
        else:
            final_centers = np.zeros(K)

        clusters = np.zeros(len(X)) #the ith index corresponds to the ith point in X. Its value is the cluster it belongs to (if clusters[i] = 2, that means the point is in cluster 2)

        cluster_sizes = np.zeros(K) #the ith index corresponds to the ith cluster center. Its value is the size of the cluster

        for i in range(len(X)):
            closest_center = nearest_center(init_centers, X[i])
            clusters[i] = closest_center
            cluster_sizes[closest_center] += 1

            if hasattr(X[0], "__len__"):
                for j in range(len(X[0])):
                    final_centers[closest_center][j] += X[i][j]
            else:
                final_centers[closest_center] += X[i]

        for i in range(K):
            if hasattr(X[0], "__len__"):
                for j in range(len(X[0])):
                    final_centers[i][j] /= cluster_sizes[i]
            else:
                final_centers[i] /= cluster_sizes[i]


    return final_centers
        
# test
# Xtrain = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [2, 4], [3, 3], [3, 4], [4, 4]]
# print(K_Means(Xtrain, 2, []))