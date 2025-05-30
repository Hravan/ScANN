import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from product_quantizer import ProductQuantizer


class ScANN:
    def __init__(self, num_clusters, top_clusters, top_k, use_pq=False, pq_num_subvectors=4, pq_num_centroids=256):
        self.num_clusters = num_clusters
        self.top_clusters = top_clusters
        self.top_k = top_k
        self.use_pq = use_pq
        self.kmeans = None
        self.data = None
        self.labels = None
        self.centroids = None

        if self.use_pq:
            self.pq = ProductQuantizer(num_subvectors=pq_num_subvectors, num_centroids=pq_num_centroids)
            self.pq_codes = {}

    def fit(self, data):
        self.data = data
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=2137)
        self.kmeans.fit(data)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

        if self.use_pq:
            self.pq.fit(data)
            codes = self.pq.encode(data)
            for i in range(self.num_clusters):
                idx = np.where(self.labels == i)[0]
                self.pq_codes[i] = (idx, codes[idx])
    
    def search(self, query):
        dists_to_centroids = euclidean_distances([query], self.centroids)[0]
        top_cluster_ids = np.argsort(dists_to_centroids)[:self.top_clusters]

        all_results = []
        for cluster_id in top_cluster_ids:
            if self.use_pq:
                candidate_indices, codes = self.pq_codes[cluster_id]
                dists = self.pq.adc_distance(query, codes)
            else:
                candidate_indices = np.where(np.isin(self.labels, top_cluster_ids))[0]
                candidates = self.data[candidate_indices]
                dists = euclidean_distances([query], candidates)[0]
            
            for index, distance in zip(candidate_indices, dists):
                all_results.append((distance, index))

        all_results.sort()
        top_indices = [i for _, i in all_results[:self.top_k]]
        return self.data[top_indices]
