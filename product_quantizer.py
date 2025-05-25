import numpy as np
from sklearn.cluster import KMeans


class ProductQuantizer:
    def __init__(self, num_subvectors=4, num_centroids=256):
        self.n_subvectors = num_subvectors  # m
        self.n_centroids = num_centroids    # k
        self.codebooks = []
    
    def fit(self, data):
        dimensionality = data.shape[1]
        self.subvector_dim, reminder = divmod(dimensionality, self.n_subvectors)
        assert not reminder, 'Vector dimension must be divisible by number of subvectors'

        for i in range(self.n_subvectors):
            subspace = self._create_subspace(data, i)
            kmeans = KMeans(n_clusters=self.n_centroids)
            kmeans.fit(subspace)
            self.codebooks.append(kmeans.cluster_centers_)
    
    def encode(self, data):
        codes = []
        for i in range(self.n_subvectors):
            subspace = self._create_subspace(data, i)
            centroids = self.codebooks[i]
            distances = self._compute_distances(subspace, centroids)
            # argmin, because the nearest centroid
            codes.append(np.argmin(distances, axis=1))
        return np.stack(codes, axis=1)
    
    def adc_distance(self, query, codes):
        # cumulative approximate distance from the query vector to each compressed (encoded) vector
        # initialized using codes.shape[0] because for each encoded vector there will be one distance overall,
        # not a distance per subvector
        distances = np.zeros(codes.shape[0])
        for i in range(self.n_subvectors):
            q_sub = query[i * self.subvector_dim:(i + 1) * self.subvector_dim]
            centroids = self.codebooks[i]
            dists = np.linalg.norm(centroids - q_sub, axis=1)
            distances += dists[codes[:, i]]
        return distances

    def _create_subspace(self, data, i):
        return data[:, i * self.subvector_dim:(i + 1) * self.subvector_dim]
    
    @staticmethod
    def _compute_distances(subspace, centroids):
        '''Compute distance between each vector and each centroid.
        subspace[:, None, :] causes broadcasting to duplicate vectors so that
        centroids[None, :, :] causes broadcasting to compare each vector to every centroid (duplicates the whole centroids matrix for each set of vectors)
        '''
        return np.linalg.norm(subspace[:, None, :] - centroids[None, :, :], axis=2)
    