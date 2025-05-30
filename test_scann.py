import numpy as np
from scann import ScANN

def test_scann_exact_mode():
    data = np.random.rand(100, 8)
    query = data[0]
    ann = ScANN(num_clusters=10, top_clusters=2, top_k=1, use_pq=False)
    ann.fit(data)
    neighbors = ann.search(query)
    assert neighbors.shape == (1, 8)
    assert np.allclose(neighbors[0], query, atol=1e-4)

def test_scann_with_pq():
    data = np.random.rand(100, 8)
    query = data[0]
    ann = ScANN(num_clusters=10, top_clusters=2, top_k=1, use_pq=True, pq_num_subvectors=2, pq_num_centroids=8)
    ann.fit(data)
    neighbors = ann.search(query)
    assert neighbors.shape == (1, 8)

def test_scann_multi_neighbors():
    data = np.random.rand(100, 8)
    query = data[0]
    ann = ScANN(num_clusters=5, top_clusters=2, top_k=5, use_pq=True, pq_num_subvectors=2, pq_num_centroids=8)
    ann.fit(data)
    neighbors = ann.search(query)
    assert neighbors.shape == (5, 8)

def test_scann_cluster_assignment():
    data = np.random.rand(100, 8)
    ann = ScANN(num_clusters=5, top_clusters=3, top_k=1, use_pq=False)
    ann.fit(data)
    assert len(np.unique(ann.labels)) == 5
    assert ann.centroids.shape == (5, 8)
