import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class ScANN:
    def __init__(self, num_clusters, top_clusters, top_k):
        self.num_clusters = num_clusters
        self.top_clusters = top_clusters
        self.top_k = top_k
        self.kmeans = None
        self.data = None
        self.labels = None
        self.centroids = None

    def fit(self, data):
        self.data = data
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=2137)
        self.kmeans.fit(data)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_
    
    def search(self, query):
        dists_to_centroids = euclidean_distances([query], self.centroids)[0]
        top_cluster_ids = np.argsort(dists_to_centroids)[:self.top_clusters]

        candidate_indices = np.where(np.isin(self.labels, top_cluster_ids))[0]
        candidates = self.data[candidate_indices]

        dists_to_candidates = euclidean_distances([query], candidates)[0]
        top_k_indices = np.argsort(dists_to_candidates)[:self.top_k]
        nearest_vectors = candidates[top_k_indices]

        return nearest_vectors
